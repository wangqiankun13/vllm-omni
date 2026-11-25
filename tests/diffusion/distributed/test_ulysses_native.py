# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CE Ulysses transport against an independent distributed reference.

Run through the shared GPU scheduler with fresh worker processes.
"""

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = pytest.mark.local_model


def _reference(x, mode, pg):
    peers = [torch.empty_like(x) for _ in range(dist.get_world_size(pg))]
    dist.all_gather(peers, x.contiguous(), group=pg)
    rank, size = dist.get_rank(pg), len(peers)
    axis = 2 if mode == 0 else 1
    width = x.shape[axis] // size
    slices = [peer.narrow(axis, rank * width, width) for peer in peers]
    return torch.cat(slices, dim=1 if mode == 0 else 2)


def _worker(rank, init_method, world_size=4):
    from vllm_omni.diffusion.distributed import a2a_permute

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world_size, ulysses_degree=world_size)
    try:
        group = get_sp_group()
        strategy = UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)

        def exchange_qkv(q, k, v):
            return a2a_permute.ulysses_qkv_fwd_group(q, k, v, group.ulysses_group.group_name, world_size)

        checks = 0
        torch.manual_seed(1234 + rank)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream), torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16):
                for batch, sequence, heads in ((1, 17, 8), (2, 65, 16), (1, 33, 8), (1, 17, 8)):
                    # Strided inputs exercise the functional API's packing contract.
                    x = torch.randn(batch, sequence, heads, 128, device=device, dtype=dtype)[..., ::2]
                    # CE groups Q/K/V while allowing fewer K/V heads (GQA).
                    kv = x[:, :, :4]
                    gate = x + 3
                    metadata = AttentionMetadata(extra={"gate_compress": gate})
                    q, k, v, metadata, ctx = strategy.pre_attention(x, kv + 1, kv + 2, metadata)
                    torch.testing.assert_close(
                        metadata.extra["gate_compress"], _reference(gate, 0, group.ulysses_group), rtol=0, atol=0
                    )
                    torch.testing.assert_close(q, _reference(x, 0, group.ulysses_group), rtol=0, atol=0)
                    torch.testing.assert_close(k, _reference(kv + 1, 0, group.ulysses_group), rtol=0, atol=0)
                    torch.testing.assert_close(v, _reference(kv + 2, 0, group.ulysses_group), rtol=0, atol=0)
                    retained = q.clone()
                    arbitrary_output = torch.randn_like(q)
                    result = strategy.post_attention(arbitrary_output, ctx)
                    torch.testing.assert_close(
                        result, _reference(arbitrary_output, 1, group.ulysses_group), rtol=0, atol=0
                    )
                    torch.testing.assert_close(q, retained, rtol=0, atol=0)
                    checks += 6
            # MiniMax VSA repeats QKV -> gate -> O throughout denoising.
            # Keep all three plans and their captured Graphs alive between cycles.
            previous: tuple[tuple[a2a_permute._CEPlan, torch.cuda.CUDAGraph | None], ...] = ()
            for step in range(3):
                incoming = x + step
                gate = incoming + 3
                metadata = AttentionMetadata(extra={"gate_compress": gate})
                q, _, _, metadata, ctx = strategy.pre_attention(incoming, kv + 1, kv + 2, metadata)
                result = strategy.post_attention(q, ctx)
                torch.testing.assert_close(result, incoming, rtol=0, atol=0)
                torch.testing.assert_close(
                    metadata.extra["gate_compress"], _reference(gate, 0, group.ulysses_group), rtol=0, atol=0
                )
                workspace = a2a_permute._SYMM_WORKSPACES[(device, group.ulysses_group.group_name)]
                plans = tuple(workspace.ce_plans.values())
                assert len(plans) == 3 and all(plan.graph is not None for plan in plans)
                if step:
                    assert all(plan is old and plan.graph is graph for plan, (old, graph) in zip(plans, previous))
                previous = tuple((plan, plan.graph) for plan in plans)
                checks += 2
            # Full-graph Dynamo capture must keep the group-bound op opaque.
            compiled_exchange = torch.compile(exchange_qkv, backend="eager", fullgraph=True)
            for actual, source in zip(compiled_exchange(x, kv + 1, kv + 2), (x, kv + 1, kv + 2)):
                torch.testing.assert_close(actual, _reference(source, 0, group.ulysses_group), rtol=0, atol=0)
                checks += 1
            # Capture without nesting the transport-local graph, then replay
            # with new input values.
            exchange_qkv(x, kv + 1, kv + 2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_q, _, _ = exchange_qkv(x, kv + 1, kv + 2)
                captured_round_trip = a2a_permute.ulysses_o_rev(captured_q, group.ulysses_group.group_name, world_size)
            x.add_(1)
            graph.replay()
            torch.testing.assert_close(captured_q, _reference(x, 0, group.ulysses_group), rtol=0, atol=0)
            torch.testing.assert_close(captured_round_trip, x, rtol=0, atol=0)
            checks += 2
            print(f"native CE rank={rank}: {checks} exact reference/lifetime checks passed", flush=True)
    finally:
        a2a_permute.clear_a2a_permute_workspaces()
        destroy_distributed_env()


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_native_ulysses_transport(tmp_path):
    if torch.accelerator.device_count() < 4:
        pytest.skip("Requires four scheduler-allocated NVLink GPUs")
    torch.multiprocessing.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4)


def _split_worker(rank, init_method, world):
    from dataclasses import replace

    from vllm_omni.diffusion.attention.backends.abstract import PackedPaddingMetadata
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.head_split import HeadSplitAttention, HeadSplitTrials, head_partitions
    from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world, ulysses_degree=world)
    try:
        cfg = OmniDiffusionConfig.from_kwargs(
            enforce_eager=True,
            parallel_config=DiffusionParallelConfig(ulysses_degree=world, ulysses_a2a_permute=True),
            diffusion_attention_backend="FLASH_ATTN",
        )
        with set_current_diffusion_config(cfg), set_forward_context(omni_diffusion_config=cfg), torch.inference_mode():
            shard, heads, dim = 64, 8 * world, 64
            layer = Attention(heads, dim, False, dim**-0.5)
            pg = get_sp_group().ulysses_group
            torch.manual_seed(1024)
            prefix = torch.randn(1, 13, heads, dim * 3, device=device, dtype=torch.bfloat16).chunk(3, dim=-1)
            torch.manual_seed(42 + rank)
            q, k, v = torch.randn(1, shard, heads, dim * 3, device=device, dtype=torch.bfloat16).chunk(3, dim=-1)
            rows = shard * world
            cu = torch.tensor([0, rows - 8, rows], dtype=torch.int32, device=device)
            packed = AttentionMetadata(
                packed_padding=PackedPaddingMetadata(rows - 8, rows - 8, cu[:2], cu[:2]),
                extra=dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=rows - 8, max_seqlen_k=rows - 8),
            )
            decode = AttentionMetadata(joint_query=prefix[0][:, :0], joint_key=prefix[1], joint_value=prefix[2])
            prefill = replace(decode, joint_query=prefix[0], full_attn_spans=[[(0, 13), (13, 13 + rows)]])
            for ratio, metadata in (("3:5", decode), ("1:6:1", packed), ("auto", prefill)):
                layer.head_split = HeadSplitAttention(head_partitions(ratio, heads, world), pg)
                with set_forward_context(omni_diffusion_config=cfg):
                    previous = None
                    calls = 4 * len(layer.head_split.candidates) + 1 if ratio == "auto" else 2
                    for i in range(calls):
                        incoming = q + i / 10
                        md = decode if ratio == "auto" and i == calls - 1 else metadata
                        expected = layer._forward_impl(incoming, k, v, replace(md))
                        actual = layer(incoming, k, v, replace(md))
                        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
                        if previous is not None:
                            torch.testing.assert_close(previous[0], previous[1], rtol=0, atol=0)
                        previous = actual, actual.clone()
                    if ratio == "auto":
                        winner = get_forward_context().head_split_trials.winner
                        assert winner is not None
                        peers = [None] * world
                        dist.all_gather_object(peers, winner, group=pg)
                        assert all(w == winner for w in peers)
            assert get_forward_context().head_split_trials is None
            # Capture once, then replay with new Q and request-owned K/V at the
            # same addresses in the graph. Reusing a stale prefix changes the result.
            controller = HeadSplitAttention(head_partitions("3:5", heads, world), pg)
            layer.head_split = controller

            def block(hidden, *, kv_cache, cache_branch, key, value):
                kv = kv_cache[0][cache_branch]
                md = replace(decode, joint_key=kv["key"], joint_value=kv["value"])
                return layer(hidden, key, value, md) + 1

            first_graph = None
            for offset, length in ((0, 13), (2, 13), (1, 9), (3, 13)):
                pk, pv = prefix[1][:, :length] + offset, prefix[2][:, :length] - offset
                cache = [{"cond": {"key": pk, "value": pv}}]
                with set_forward_context(omni_diffusion_config=cfg):
                    get_forward_context().head_split_trials = HeadSplitTrials(controller.candidates)
                    for step in (1, 2):
                        incoming = q + step
                        expected = layer._forward_impl(incoming, k, v, replace(decode, joint_key=pk, joint_value=pv))
                        with controller.blocks.request_stream():
                            actual = controller.run_blocks(
                                block, incoming, {"key": k, "value": v, "cache_branch": "cond"}, kv_cache=cache
                            )
                        torch.testing.assert_close(actual, expected + 1, atol=0.01, rtol=0.01)
                        entry = next(iter(controller.blocks.entries.values()))
                        if first_graph is None:
                            first_graph = entry[0]
                        if offset in (0, 2):
                            assert entry[0] is first_graph
                        else:
                            assert entry[0] is not first_graph
            # The bound is on the global sequence, independent of the SP degree.
            shard_over_limit = 8576 // world + 1
            oversized = q[:, :1].expand(-1, shard_over_limit, -1, -1)
            for ratio in ("3:5", "auto"):
                with set_forward_context(omni_diffusion_config=cfg):
                    controller = HeadSplitAttention(head_partitions(ratio, heads, world), pg)
                    layer.head_split = controller
                    with pytest.raises(ValueError, match="8576 global packed rows"):
                        layer(oversized, oversized, oversized, packed)
                    assert controller.graphs is None
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(2, marks=hardware_marks(res={"cuda": "H100"}, num_cards=2)),
        pytest.param(4, marks=hardware_marks(res={"cuda": "H100"}, num_cards=4)),
    ],
)
def test_head_split_attention(tmp_path, world):
    if torch.accelerator.device_count() < world:
        pytest.skip(f"Requires {world} peer-accessible CUDA GPUs with FlashAttention-3")
    torch.multiprocessing.spawn(_split_worker, args=(f"file://{tmp_path / 'rendezvous'}", world), nprocs=world)
