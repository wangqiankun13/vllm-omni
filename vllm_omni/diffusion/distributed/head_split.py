# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-scoped selection of head partitions for strict Ulysses attention."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import median
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.platforms.cuda.ulysses_head_split import HeadSplitGraphs

logger = init_logger(__name__)

# Largest packed H3 sequence validated end to end. A larger wide-video
# request produced a divergent cold result with fixed splitting.
_H3_MAX_PACKED_ROWS = 8576


def parse_head_split(spec: str) -> tuple[int, ...]:
    if spec == "auto":
        return ()
    fields = spec.split(":")
    if len(fields) not in (2, 3) or any(not f.isascii() or not f.isdecimal() or int(f) == 0 for f in fields):
        raise ValueError("ulysses_head_split must be 'auto' or a ratio of two or three positive integers, e.g. 3:5")
    weights = tuple(map(int, fields))
    divisor = math.gcd(*weights)
    return tuple(w // divisor for w in weights)


def head_partitions(spec: str, heads: int, world: int, *, packed: bool = False) -> tuple[tuple[int, ...], ...]:
    if world not in (2, 4) or heads <= 0 or heads % world:
        raise ValueError(f"ulysses_head_split requires SP2 or SP4 evenly dividing the {heads} attention heads")
    local = heads // world
    ratio = parse_head_split(spec)
    if ratio:
        factor, remainder = divmod(local, sum(ratio))
        if remainder:
            raise ValueError(f"ulysses_head_split={spec} cannot split {local} local heads at SP{world}")
        return (tuple(w * factor for w in ratio),)
    if local < 2:
        raise ValueError("automatic head splitting requires at least two heads per rank")
    half, edge, first = local // 2, max(2, local // 6), max(1, round(3 * local / 8))
    # Empty tuple denotes the official whole-head implementation. The one-group
    # direct exchange is also eligible: it removes copies but has no overlap.
    candidates: tuple[tuple[int, ...], ...] = (
        (),
        (local,),
        (half, local - half),
        (1, local - 2, 1),
        (edge, local - 2 * edge, edge),
        (first, local - first),
    )
    if packed:
        # The packed H3 trial also evaluates each layer's native output. Keep
        # balanced, asymmetric and short-tail overlap candidates in that budget.
        candidates = ((), (half, local - half), (1, local - 2, 1), (first, local - first))
    return tuple(dict.fromkeys(parts for parts in candidates if all(w > 0 for w in parts)))


class HeadSplitTrials:
    def __init__(self, candidates: tuple[tuple[int, ...], ...]):
        self.candidates = candidates
        self.samples: list[tuple[torch.Event, torch.Event, torch.Tensor]] = []
        self.winner: tuple[int, ...] | None = candidates[0] if len(candidates) == 1 else None

    def measure(self, run, reference, pg, device, *, timed_run=None):
        width = self.candidates[len(self.samples) // 4]
        start, end = (torch.Event(device=device, enable_timing=True) for _ in range(2))
        if timed_run is None:
            start.record()
            out = run(width)
            end.record()
        else:
            out = timed_run(width, start, end)
        if reference is None:
            reference = run(()) if width else out
        mismatch = (out != reference).any() if width else torch.zeros((), device=device, dtype=torch.bool)
        self.samples.append((start, end, mismatch))
        if len(self.samples) == 4 * len(self.candidates):
            end.synchronize()
            times = torch.tensor([a.elapsed_time(b) for a, b, _ in self.samples], device=device, dtype=torch.float32)
            dist.all_reduce(times, op=dist.ReduceOp.MAX, group=pg)
            rejected = torch.stack([m for _, _, m in self.samples]).reshape(-1, 4).any(dim=1).int()
            dist.all_reduce(rejected, op=dist.ReduceOp.MAX, group=pg)
            values = times.tolist()
            # The first call of each candidate can initialize/capture a graph.
            scores = [median(values[i + 1 : i + 4]) for i in range(0, len(values), 4)]
            eligible = [i for i, failed in enumerate(rejected.tolist()) if not failed]
            best = min(eligible, key=scores.__getitem__)
            self.winner = self.candidates[best] if scores[best] < 0.99 * scores[0] else ()
            if dist.get_rank(pg) == 0:
                logger.info(
                    "Ulysses head split selected %s, slowest-rank median ms %s",
                    self.winner or "native",
                    dict(zip(self.candidates, scores)),
                )
            self.samples.clear()
        return reference


@dataclass(frozen=True)
class _TargetLayout:
    prefix: tuple[torch.Tensor, ...]
    boundaries: tuple[torch.Tensor, ...]
    maxima: tuple[int, int]
    prefill: bool


def _target_layout(layer, q, k, v, md: AttentionMetadata | None, world: int) -> _TargetLayout:
    if (
        not q.is_cuda
        or q.dtype != torch.bfloat16
        or q.ndim != 4
        or q.shape[0] != 1
        or any(x.shape != q.shape or x.dtype != q.dtype or x.device != q.device for x in (k, v))
    ):
        raise ValueError("ulysses_head_split requires batch-one BF16 MHA with equal Q/K/V shapes on CUDA")
    if (
        md is None
        or md.attn_mask is not None
        or md.joint_attn_mask is not None
        or md.query_ranges is not None
        or any(key in md.extra for key in ("gate_compress", "vdn_window", "minimax_h3_reference_kv_post_parallel"))
    ):
        raise ValueError("ulysses_head_split requires unmasked Qwen target or MiniMax-H3 packed attention")
    prefix: tuple[torch.Tensor, ...] = ()
    boundaries: tuple[torch.Tensor, ...] = ()
    shard = q.size(1)
    rows = shard * world
    prefill = False
    if md.joint_key is not None:
        if md.joint_value is None:
            raise ValueError("ulysses_head_split needs both joint key and value")
        prefix = (md.joint_key, md.joint_value)
        if (
            md.joint_query is None
            or md.joint_strategy != "front"
            or any(
                p is None or p.shape[0] != 1 or p.shape[2:] != q.shape[2:] or p.dtype != q.dtype or p.device != q.device
                for p in (*prefix, md.joint_query)
            )
            or prefix[0].shape != prefix[1].shape
        ):
            raise ValueError("ulysses_head_split requires a replicated front prefix with matching K/V")
        length = prefix[0].size(1)
        prefill = md.joint_query.size(1) > 0
        if prefill and not (
            md.joint_query.size(1) == length
            and md.full_attn_spans
            and len(md.full_attn_spans) == 1
            and md.full_attn_spans[0]
            and md.full_attn_spans[0][-1] == (length, length + rows)
            and all(end <= length for _, end in md.full_attn_spans[0][:-1])
        ):
            raise ValueError("ulysses_head_split requires Qwen block-causal prefill with a full target span")
        maxima = (rows, rows + length)
    elif md.packed_padding is not None:
        if rows > _H3_MAX_PACKED_ROWS:
            raise ValueError(
                f"ulysses_head_split on MiniMax-H3 supports up to {_H3_MAX_PACKED_ROWS} global packed rows, "
                f"got {rows}. Omit --ulysses-head-split for this request size."
            )
        boundaries = tuple(md.extra[name] for name in ("cu_seqlens_q", "cu_seqlens_k"))
        maxima = (md.extra["max_seqlen_q"], md.extra["max_seqlen_k"])
        if (
            any(
                b.ndim != 1 or b.numel() not in (2, 3) or b.dtype != torch.int32 or b.device != q.device
                for b in boundaries
            )
            or max(maxima) > rows
        ):
            raise ValueError("ulysses_head_split requires a single packed H3 request with equal SP shards")
    else:
        raise ValueError("ulysses_head_split requires a Qwen prefix or H3 packed sequence lengths")
    if not prefill and md.full_attn_spans is not None:
        raise ValueError("ulysses_head_split supports piecewise attention only during Qwen prefill")

    if q.size(2) != layer.num_heads or layer._get_active_parallel_strategy() is not layer.parallel_strategy:
        raise ValueError("ulysses_head_split requires active SP and the configured number of heads")
    return _TargetLayout(prefix, boundaries, maxima, prefill)


class HeadSplitAttention:
    """One geometry cache per loaded DiT. The timing decision belongs to its request."""

    def __init__(self, candidates, pg):
        from vllm_omni.platforms.cuda.ulysses_head_split import HeadSplitBlockGraphs

        self.blocks = HeadSplitBlockGraphs(pg)
        self.candidates = candidates
        self.pg = pg
        self.world = dist.get_world_size(pg)
        self.in_block_graph = False
        self.signature = None
        self.graphs: HeadSplitGraphs | None = None

    def run_blocks(self, forward, hidden, kwargs, *, kv_cache=None):
        from vllm_omni.diffusion.distributed.a2a_permute import _SYMM_WORKSPACES
        from vllm_omni.diffusion.forward_context import get_forward_context

        ctx = get_forward_context()
        trials = ctx.head_split_trials
        if trials is None or trials.winner is None or ctx.denoise_step_idx == 0:
            if kv_cache is not None:
                kwargs = dict(kwargs, kv_cache=kv_cache)
            return forward(hidden, **kwargs)

        def keep_transport():
            # The official CE workspace can grow on a later request. Its old
            # peer pointers remain embedded in any resident native-route graph.
            return self.graphs, _SYMM_WORKSPACES.get((hidden.device, self.pg.group_name))

        self.in_block_graph = True
        try:
            return self.blocks.run(forward, hidden, kwargs, trials, keep_transport, kv_cache=kv_cache)
        finally:
            self.in_block_graph = False

    def __call__(self, layer: Attention, q, k, v, md: AttentionMetadata | None) -> torch.Tensor:
        from vllm_omni.diffusion.forward_context import get_forward_context
        from vllm_omni.platforms.cuda.ulysses_head_split import HeadSplitGraphs

        ctx = get_forward_context()
        trials = ctx.head_split_trials
        if trials is None:
            # A rank-local validation error must not strand peers in their first
            # rendezvous, or let them compute remote offsets from different shapes.
            try:
                layout = _target_layout(layer, q, k, v, md, self.world)
                geometry = [
                    *q.shape,
                    *layout.maxima,
                    int(layout.prefill),
                    len(layout.prefix),
                    *(tuple(b.numel() for b in layout.boundaries) or (0, 0)),
                ]
                reason = None
            except ValueError as error:
                geometry, reason = [-1] * 10, str(error)
            local = torch.tensor(geometry, device=q.device)
            peers = local.new_empty(self.world * len(geometry))
            dist.all_gather_into_tensor(peers, local, group=self.pg)
            if reason is not None or not torch.equal(peers.view(self.world, -1), local.expand(self.world, -1)):
                raise ValueError(reason or "ulysses_head_split requires equal supported geometry on every SP rank")
            trials = ctx.head_split_trials = HeadSplitTrials(self.candidates)
        else:
            layout = _target_layout(layer, q, k, v, md, self.world)
        prefix, boundaries, maxima = layout.prefix, layout.boundaries, layout.maxima
        assert md is not None  # validated by _target_layout on all ranks
        prefill = layout.prefill
        probing = trials.winner is None
        # Prefill rows must keep their block-causal semantics. Only its target
        # queries see the same K/V as decode, so use them for timing.
        if prefill:
            output = layer._forward_impl(q, k, v, replace(md))
            if not probing:
                return output
            assert md.joint_query is not None
            target_md = replace(md, full_attn_spans=None, joint_query=md.joint_query[:, :0])
        else:
            target_md = md

        def transport():
            signature = (
                q.device,
                tuple(q.shape),
                tuple(p.stride(1) for p in (q, k, v)),
                tuple(p.shape for p in prefix),
                maxima,
                tuple(b.numel() for b in boundaries),
                layer.softmax_scale,
            )
            if signature != self.signature:
                if self.graphs is not None:
                    # Peer pointers in the previous graph must be idle on *all*
                    # ranks before any process can release its storage.
                    torch.accelerator.synchronize(q.device)
                    dist.barrier(group=self.pg)
                    self.signature = None
                    self.graphs = None
                self.graphs = HeadSplitGraphs((q, k, v), prefix, boundaries, maxima, layer.softmax_scale, self.pg)
                self.signature = signature
            assert self.graphs is not None
            return self.graphs

        def run(widths):
            if not widths:
                if probing and boundaries:
                    # timed_packed already staged these exact inputs and ran
                    # the native candidate first. Reuse that graph for the
                    # accuracy comparison, outside the candidate's timer.
                    return transport().packed_reference()
                return layer._forward_impl(q, k, v, replace(target_md))
            return transport().run(widths, (q, k, v), prefix, boundaries, in_block_graph=self.in_block_graph)

        if probing:
            # Trials cannot change the state that subsequent layers observe.
            reference = output[:, -q.size(1) :] if prefill else None

            def timed_packed(widths, start, end):
                return transport().timed_packed(widths, (q, k, v), target_md, layer, start, end)

            result = trials.measure(run, reference, self.pg, q.device, timed_run=timed_packed if boundaries else None)
        else:
            result = run(trials.winner)
        return output if prefill else result


def configure_head_split(pipeline, config):
    name = type(pipeline).__name__
    if name == "MiniMaxH3Pipeline" and config.parallel_config.ulysses_degree != 4:
        raise ValueError("ulysses_head_split on MiniMax-H3 requires SP4. Omit --ulysses-head-split for H3 at SP2.")

    from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func, vllm_varlen_version_kwargs
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    from vllm_omni.platforms import current_omni_platform

    if (
        not config.enforce_eager
        or config.step_execution
        or config.streaming_output
        or config.diffusion_use_graph
        or config.diffusion_attention_schedule is not None
        or config.cache_backend not in (None, "none")
    ):
        raise ValueError(
            "ulysses_head_split requires request-level --enforce-eager without denoise graphs, "
            "cache acceleration or attention schedules"
        )
    if config.dtype != torch.bfloat16:
        raise ValueError("ulysses_head_split requires BF16 attention inputs")
    if config.fa_deterministic:
        raise ValueError("ulysses_head_split does not support fa_deterministic")
    if not current_omni_platform.is_cuda() or vllm_varlen_version_kwargs(flash_attn_varlen_func) != {"fa_version": 3}:
        raise ValueError("ulysses_head_split requires CUDA and the vLLM FlashAttention-3 varlen backend")
    from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, should_offload_component

    if (
        should_offload_component(config, DIT_COMPONENT)
        or config.enable_sleep_mode
        or config.lora_path
        or config.weight_transfer_config
    ):
        raise ValueError("ulysses_head_split block graphs require stationary DiT weights without LoRA or offload")
    if name == "QwenImage21Pipeline":
        layers = [block.attn.attn for block in pipeline.transformer.transformer_blocks]
    elif name == "MiniMaxH3Pipeline" and str(config.task_type).lower() == "t2va":
        pipeline.supported_tasks = pipeline.supported_tasks.intersection({"t2va"})
        layers = [block.attn.attention for block in pipeline.transformer.blocks]
    else:
        raise ValueError("ulysses_head_split supports Qwen-Image-2.1 and MiniMax-H3 with task_type=t2va")
    from vllm.distributed.parallel_state import in_the_same_node_as

    sp = get_sp_group()
    if not all(in_the_same_node_as(sp.cpu_group, source_rank=0)):
        raise ValueError("ulysses_head_split requires all SP ranks on one CUDA peer-accessible node")
    pg = sp.ulysses_group
    world = dist.get_world_size(pg)
    if world != config.parallel_config.ulysses_degree:
        raise ValueError("ulysses_head_split process group does not match configured degree")
    heads = layers[0].num_heads
    candidates = head_partitions(
        config.parallel_config.ulysses_head_split, heads, world, packed=name == "MiniMaxH3Pipeline"
    )
    if len(candidates) > 1 and len(layers) < 4 * len(candidates):
        raise ValueError("ulysses_head_split=auto needs four first-step layers per candidate")
    controller = HeadSplitAttention(candidates, pg)
    pipeline.transformer.head_split_controller = controller
    for layer in layers:
        if (
            layer.num_heads != heads
            or layer.num_kv_heads != heads
            or layer.causal
            or layer.use_ring
            or layer.attn_backend.get_name() != "FLASH_ATTN"
            or layer._effective_kv_cache_dtype() not in (None, "float")
            or layer.paged_kv_cache_role is not None
            or layer.scatter_idx != 2
            or layer.gather_idx != 1
        ):
            raise ValueError("ulysses_head_split requires homogeneous unquantized, noncausal FLASH_ATTN MHA layers")
        layer.head_split = controller
    if dist.get_rank(pg) == 0:
        logger.info(
            "Enabled Ulysses head split %s on %d %s DiT layers (SP%d, %d heads per rank)",
            config.parallel_config.ulysses_head_split,
            len(layers),
            name,
            world,
            heads // world,
        )
    return controller
