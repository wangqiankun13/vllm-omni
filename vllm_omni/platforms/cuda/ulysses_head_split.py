# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph-replayed CUDA copy-engine exchange for batch-one strict Ulysses."""

from contextlib import contextmanager
from dataclasses import replace

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func
from vllm_omni.diffusion.distributed.a2a_permute import _ce_runtime


class HeadSplitGraphs:
    """Share staging and one symmetric allocation across ratios of the same geometry."""

    def __init__(self, parts, prefix, boundaries, maxima, scale, pg):
        self.pg = pg
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
        self.trial_graphs = {}
        self.runtime = _ce_runtime()
        self.inputs = tuple(
            torch.empty_strided(p.shape, p.stride(), dtype=p.dtype, device=p.device)
            if boundaries and p.stride(1) == 3 * heads * dim and p.stride()[2:] == (dim, 1)
            else torch.empty_like(p, memory_format=torch.contiguous_format)
            for p in parts
        )
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

    def _exchange(self, widths, inputs, prefix, boundaries, owner):
        assert flash_attn_varlen_func is not None  # checked before installing the split policy
        q = inputs[0]
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
                        slot[:prefix_len].copy_(prefix[index - 1][0, :, head_offset : head_offset + width])
                    offsets.append(slot_offset * q.element_size() + (prefix_len + self.rank * shard) * pitch)
                    slot_offset += count
                for order in range(1, self.world + 1):
                    peer = self.rank ^ (order % self.world)
                    src_offset = (peer * self.local_heads + head_offset) * head_bytes
                    for source, dst_offset in zip(inputs, offsets, strict=True):
                        dst = self.handle.buffer_ptrs[peer] + dst_offset
                        self.copy(
                            self.send,
                            dst,
                            pitch,
                            source.data_ptr() + src_offset,
                            source.stride(1) * q.element_size(),
                            pitch,
                            shard,
                        )
                self.handle.barrier()
            owner.wait_stream(self.send)
            attended = flash_attn_varlen_func(
                q=slots[0],
                k=slots[1],
                v=slots[2],
                cu_seqlens_q=boundaries[0],
                cu_seqlens_k=boundaries[1],
                max_seqlen_q=self.maxima[0],
                max_seqlen_k=self.maxima[1],
                softmax_scale=self.scale,
                causal=False,
                fa_version=3,
                # The verified H3 two-way routes use the FA3 heuristic. Keep
                # the pinned reduction for Qwen and other fixed partitions.
                num_splits=0 if not self.prefix_len and widths in ((7, 7), (5, 9)) else 1,
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

    def packed_reference(self):
        graph, output, _ = self.trial_graphs[()]
        graph.replay()
        return output.clone()

    def timed_packed(self, widths, parts, metadata, layer, start, end):
        """Time the same graph boundary for native and split H3 candidates."""
        from vllm_omni.diffusion.distributed.a2a_permute import _SYMM_WORKSPACES

        for dst, src in zip(self.inputs, parts, strict=True):
            dst.copy_(src)
        for key, dst in zip(("cu_seqlens_q", "cu_seqlens_k"), self.boundaries, strict=True):
            dst.copy_(metadata.extra[key])
        # If a later sampled layer introduces a new geometry, also capture
        # its native reference before comparing that layer's candidate.
        for route in ((), widths):
            if route in self.trial_graphs:
                continue
            q_cu, k_cu = self.boundaries
            md = replace(
                metadata,
                extra=dict(metadata.extra, cu_seqlens_q=q_cu, cu_seqlens_k=k_cu),
                packed_padding=replace(metadata.packed_padding, cu_seqlens_q=q_cu[:2], cu_seqlens_k=k_cu[:2]),
            )

            def forward(route=route, md=md):
                if not route:
                    return layer._forward_impl(*self.inputs, replace(md))
                self._exchange(route, self.inputs, (), self.boundaries, self.owner)
                return self.output

            forward()
            self.owner.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.owner):
                output = forward()
            workspace = _SYMM_WORKSPACES.get((self.storage.device, self.pg.group_name))
            self.trial_graphs[route] = graph, output, workspace
        graph, output, _ = self.trial_graphs[widths]
        start.record()
        graph.replay()
        end.record()
        return output.clone()

    def run(self, widths, parts, prefix, boundaries, *, in_block_graph=False):
        if in_block_graph:
            # The block graph owns live Q/K/V addresses through the reverse
            # join. Its output projection consumes the view before the next
            # block can cross the symmetric entry barrier and overwrite it.
            inputs = tuple(p if p.stride(-1) == 1 and p.stride(-2) == p.size(-1) else p.contiguous() for p in parts)
            local_prefix = tuple(p[:, :, self.prefix_heads] for p in prefix)
            self._exchange(widths, inputs, local_prefix, boundaries or self.boundaries, torch.cuda.current_stream())
            return self.output
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
                self._exchange(widths, self.inputs, self.prefix, self.boundaries, self.capture)
            self.capture.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.capture):
                self._exchange(widths, self.inputs, self.prefix, self.boundaries, self.capture)
            self.owner.wait_stream(self.capture)
            self.graphs[widths] = graph
        self.graphs[widths].replay()
        # A subsequent call can overwrite the symmetric output on every rank.
        return self.output.clone()


class HeadSplitBlockGraphs:
    """Serial replay of the decode block stack for one loaded DiT and input geometry."""

    def __init__(self, pg):
        self.pg = pg
        self.stream = torch.cuda.Stream()
        self.clear()

    @contextmanager
    def request_stream(self):
        owner = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(owner)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            owner.wait_stream(self.stream)

    def clear(self):
        self.signature = None
        self.static = {}
        self.prefix_cache = None
        self.request = None
        self.prefix_source = None
        self.entries = {}
        self.pool = None

    def run(self, forward, hidden, kwargs, trials, keep_transport, *, kv_cache=None):
        from vllm_omni.diffusion.cuda_graph import clone_static, copy_into, kwargs_signature

        # Qwen's immutable prefix is shared by all decode steps of a request.
        inputs = dict(kwargs, hidden=hidden)
        signature = (kwargs_signature(inputs), kwargs_signature(kv_cache[0]) if kv_cache else None)
        new_geometry = self.signature != signature
        if new_geometry:
            if self.entries:
                # A peer may still be reading a previous graph's symmetric allocation.
                self.stream.synchronize()
                dist.barrier(group=self.pg)
            self.clear()
            self.static = clone_static(inputs)
            self.prefix_cache = clone_static(kv_cache)
            self.signature = signature
        else:
            copy_into(self.static, inputs)
        if kv_cache is not None:
            source = kv_cache[0][kwargs["cache_branch"]]["key"]
            if self.request is not trials or self.prefix_source is not source:
                if not new_geometry:
                    copy_into(self.prefix_cache, kv_cache)
                self.request, self.prefix_source = trials, source
        key = trials.winner
        if key not in self.entries:
            args = dict(self.static)
            if kv_cache is not None:
                args["kv_cache"] = self.prefix_cache
            stream = torch.cuda.current_stream(hidden.device)
            forward(**args)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=self.pool):
                output = forward(**args)
            self.pool = graph.pool()
            self.entries[key] = graph, output, keep_transport()
        graph, output, _ = self.entries[key]
        graph.replay()
        return output
