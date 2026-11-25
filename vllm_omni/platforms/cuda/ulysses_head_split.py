# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph-replayed CUDA copy-engine exchange for batch-one strict Ulysses."""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
from vllm_omni.diffusion.distributed.a2a_permute import _ce_runtime


class HeadSplitGraphs:
    """Share staging and one symmetric allocation across ratios of the same geometry."""

    def __init__(self, parts, prefix, boundaries, maxima, scale, pg):
        self.world = dist.get_world_size(pg)
        self.rank = dist.get_rank(pg)
        q = parts[0]
        _, shard, heads, dim = q.shape
        self.local_heads = heads // self.world
        self.rows = shard * self.world
        self.prefix_len = prefix[0].size(1) if prefix else 0
        self.maxima = maxima
        self.scale = scale
        self.owner = torch.cuda.current_stream(q.device)
        self.capture = torch.cuda.Stream(device=q.device)
        self.send = torch.cuda.Stream(device=q.device)
        self.receive = torch.cuda.Stream(device=q.device)
        self.graphs = {}
        self.runtime = _ce_runtime()
        self.inputs = tuple(torch.empty_like(p, memory_format=torch.contiguous_format) for p in parts)
        # Prefix K/V are replicated. Stage only the columns consumed on this
        # rank, independent of the number of groups in the winning partition.
        self.prefix_heads = slice(self.rank * self.local_heads, (self.rank + 1) * self.local_heads)
        self.prefix = tuple(p.new_empty((1, self.prefix_len, self.local_heads, dim)) for p in prefix)
        self.boundaries = (
            tuple(b.clone() for b in boundaries)
            if boundaries
            else tuple(torch.tensor([0, length], device=q.device, dtype=torch.int32) for length in maxima)
        )
        elements = (4 * self.rows + 2 * self.prefix_len) * self.local_heads * dim
        self.storage = symm._SymmetricMemory.empty_strided_p2p((elements,), (1,), q.dtype, q.device)
        self.handle = symm.rendezvous(self.storage, pg.group_name)
        if len(self.handle.buffer_ptrs) != self.world or not all(self.handle.buffer_ptrs):
            raise ValueError("ulysses_head_split needs symmetric peer access to every GPU on the node")
        self.output = self.storage[-q.numel() :].view_as(q)

    def copy(self, stream, dst, dst_pitch, src, src_pitch, width, rows):
        status = self.runtime.lib.cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, rows, 4, stream.cuda_stream)
        self.runtime.CUDART_CHECK(status)

    def _exchange(self, widths):
        assert flash_attn_varlen_func is not None  # checked before installing the split policy
        owner = self.capture
        q = self.inputs[0]
        _, shard, heads, dim = q.shape
        head_bytes = dim * q.element_size()
        output_offset = self.output.data_ptr() - self.storage.data_ptr()
        head_offset = slot_offset = 0
        # Entry protects all peer reads from the preceding invocation. There is one
        # ordered series of barriers per handle, including the final reverse join.
        self.handle.barrier()
        self.send.wait_stream(owner)
        outputs = []
        for width in widths:
            pitch = width * head_bytes
            first = self.rank * self.local_heads + head_offset
            slots = []
            with torch.cuda.stream(self.send):
                offsets = []
                for index in range(3):
                    prefix_len = self.prefix_len if index else 0
                    length = self.rows + prefix_len
                    count = length * width * dim
                    slot = self.storage.narrow(0, slot_offset, count).view(length, width, dim)
                    slots.append(slot)
                    if prefix_len:
                        slot[:prefix_len].copy_(self.prefix[index - 1][0, :, head_offset : head_offset + width])
                    offsets.append(slot_offset * q.element_size() + (prefix_len + self.rank * shard) * pitch)
                    slot_offset += count
                for order in range(1, self.world + 1):
                    peer = self.rank ^ (order % self.world)
                    src_offset = (peer * self.local_heads + head_offset) * head_bytes
                    for source, dst_offset in zip(self.inputs, offsets, strict=True):
                        dst = self.handle.buffer_ptrs[peer] + dst_offset
                        self.copy(
                            self.send, dst, pitch, source.data_ptr() + src_offset, heads * head_bytes, pitch, shard
                        )
                self.handle.barrier()
            owner.wait_stream(self.send)
            attended = flash_attn_varlen_func(
                q=slots[0],
                k=slots[1],
                v=slots[2],
                cu_seqlens_q=self.boundaries[0],
                cu_seqlens_k=self.boundaries[1],
                max_seqlen_q=self.maxima[0],
                max_seqlen_k=self.maxima[1],
                softmax_scale=self.scale,
                causal=False,
                fa_version=3,
                num_splits=1,
            )
            if isinstance(attended, tuple):
                attended = attended[0]
            # Only the reverse stream consumes this allocation after the FA kernel.
            # Join before allocator reuse on the owner (also during warmup).
            outputs.append(attended)
            self.receive.wait_stream(owner)
            src_pitch = attended.stride(0) * q.element_size()
            for order in range(1, self.world + 1):
                peer = self.rank ^ (order % self.world)
                dst = self.handle.buffer_ptrs[peer] + output_offset + first * head_bytes
                src = attended.data_ptr() + peer * shard * src_pitch
                self.copy(self.receive, dst, heads * head_bytes, src, src_pitch, pitch, shard)
            head_offset += width
        owner.wait_stream(self.receive)
        self.handle.barrier()

    def run(self, widths, parts, prefix, boundaries):
        if self.owner != torch.cuda.current_stream(self.storage.device):
            raise RuntimeError("ulysses_head_split requires a single submission stream")
        for dst, src in zip(self.inputs, parts, strict=True):
            dst.copy_(src)
        for dst, src in zip(self.prefix, prefix, strict=True):
            dst.copy_(src[:, :, self.prefix_heads])
        for dst, src in zip(self.boundaries, boundaries):
            dst.copy_(src)
        if widths not in self.graphs:
            self.capture.wait_stream(self.owner)
            with torch.cuda.stream(self.capture):
                self._exchange(widths)
            self.capture.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.capture):
                self._exchange(widths)
            self.owner.wait_stream(self.capture)
            self.graphs[widths] = graph
        self.graphs[widths].replay()
        # A subsequent call can overwrite the symmetric output on every rank.
        return self.output.clone()
