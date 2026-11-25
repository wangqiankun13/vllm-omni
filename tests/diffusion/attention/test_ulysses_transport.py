# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("use_ce", [True, False])
def test_selected_transport_handles_qkv_and_output(monkeypatch, use_ce):
    sp = SimpleNamespace(
        ulysses_group=SimpleNamespace(group_name="subgroup"),
        ulysses_world_size=2,
        ring_world_size=1,
        allgather_world_size=1,
    )
    initialized: list[bool] = []
    exchanges: list[int] = []
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: initialized.append(True))
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default: "strict")
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: 2)

    def exchange(x, group_name, world_size, mode):
        assert group_name == "subgroup" and world_size == 2
        exchanges.append(mode)
        b, s, h, d = x.shape
        return x.new_empty((b, s * 2, h // 2, d) if mode == 0 else (b, s // 2, h * 2, d))

    def nccl_exchange(pg, x, scatter, gather, sync):
        assert not use_ce and pg is sp.ulysses_group
        return exchange(x, pg.group_name, 2, 0 if scatter == 2 else 1)

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", nccl_exchange)
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", lambda x, g, w: exchange(x, g, w, 1))

    def exchange_qkv(q, k, v, group_name, world_size):
        assert group_name == "subgroup" and world_size == 2 and use_ce
        exchanges.append(2)
        return q.new_empty(q.numel() + k.numel() + v.numel())

    monkeypatch.setattr(a2a_permute, "ulysses_qkv_ce", exchange_qkv)
    strategy = ulysses.UlyssesParallelAttention(sp, 2, 1, False, ulysses_a2a_permute=use_ce)
    q = torch.empty(1, 16, 8, 64)
    q, k, v, _, ctx = strategy.pre_attention(q, q, q, None)
    result = strategy.post_attention(q, ctx)
    assert initialized == ([True] if use_ce else [])
    assert exchanges == ([2, 1] if use_ce else [0, 0, 0, 1])
    assert result.shape == (1, 16, 8, 64)


@pytest.mark.parametrize("scatter,gather,size", [(1, 2, 2), (2, 1, 1)])
def test_ineligible_layout_does_not_initialize_ce(monkeypatch, scatter, gather, size):
    sp = SimpleNamespace(ulysses_group=None, ulysses_world_size=size, ring_world_size=1, allgather_world_size=1)
    monkeypatch.setattr(
        a2a_permute, "ensure_a2a_permute_available", lambda: pytest.fail("Unexpected CE initialization")
    )
    strategy = ulysses.UlyssesParallelAttention(sp, scatter, gather, False, ulysses_a2a_permute=True)
    assert not ulysses._a2a_permute_enabled(strategy._ulysses_a2a_permute, scatter, gather, size)


@pytest.mark.parametrize(
    "world,heads,ratio,widths",
    [
        (2, 32, "3:5", (6, 10)),
        (4, 32, "3:5", (3, 5)),
        (4, 56, "1:12:1", (1, 12, 1)),
        (2, 56, "2:2", (14, 14)),
    ],
)
def test_head_split_partitions(world, heads, ratio, widths):
    from vllm_omni.diffusion.distributed.head_split import head_partitions

    assert head_partitions(ratio, heads, world) == (widths,)
    native, *candidates = head_partitions("auto", heads, world)
    assert native == ()
    assert all(sum(candidate) == heads // world and min(candidate) > 0 for candidate in candidates)


@pytest.mark.parametrize(
    "world,heads,ratio",
    [
        (1, 32, "auto"),
        (8, 32, "auto"),
        (4, 30, "1:1"),
        (4, 32, "1:2"),
        (2, 32, "1:0"),
        (2, 32, "-1:3"),
        (2, 32, "1:2:3:4"),
        (2, 32, "fastest"),
    ],
)
def test_head_split_invalid(world, heads, ratio):
    from vllm_omni.diffusion.distributed.head_split import head_partitions

    with pytest.raises(ValueError, match="ulysses_head_split"):
        head_partitions(ratio, heads, world)


def test_packed_selector_keeps_balanced_asymmetric_and_short_tail_trials():
    from vllm_omni.diffusion.distributed.head_split import head_partitions

    assert head_partitions("auto", 56, 4, packed=True) == ((), (7, 7), (1, 12, 1), (5, 9))
    assert head_partitions("2:10:2", 56, 4, packed=True) == ((2, 10, 2),)


def test_head_split_config_is_opt_in():
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    assert DiffusionParallelConfig().ulysses_head_split is None
    cfg = dict(ulysses_degree=2, ulysses_a2a_permute=True, ulysses_head_split="auto")
    assert DiffusionParallelConfig(**cfg).ulysses_head_split == "auto"
    for overrides in (
        {"ulysses_degree": 8},
        {"ulysses_a2a_permute": False},
        {"ring_degree": 2},
        {"ulysses_mode": "advanced_uaa"},
        {"tensor_parallel_size": 2},
    ):
        with pytest.raises(ValueError, match="ulysses_head_split"):
            DiffusionParallelConfig(**(cfg | overrides))


@pytest.mark.parametrize("ratio", ["5:9", "auto"])
def test_h3_head_split_requires_sp4(ratio):
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.distributed.head_split import configure_head_split

    pipeline = type("MiniMaxH3Pipeline", (), {})()
    parallel = DiffusionParallelConfig(ulysses_degree=2, ulysses_a2a_permute=True, ulysses_head_split=ratio)
    with pytest.raises(ValueError, match="MiniMax-H3 requires SP4"):
        configure_head_split(pipeline, SimpleNamespace(parallel_config=parallel))


@pytest.mark.parametrize(
    "remote_ms,remote_mismatch,expected", [(6.0, False, (1, 1)), (9.95, False, ()), (20.0, False, ()), (6.0, True, ())]
)
def test_automatic_head_split_uses_critical_rank_and_native_output(monkeypatch, remote_ms, remote_mismatch, expected):
    from vllm_omni.diffusion.distributed import head_split

    times = iter([100, 10, 10, 10, 100, 3, 3, 3])

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return next(times)

    def reduce(tensor, op, group):
        assert op == head_split.dist.ReduceOp.MAX and group is pg
        if tensor.is_floating_point():
            tensor[4:].clamp_(min=remote_ms)
        else:
            tensor[1] |= int(remote_mismatch)

    pg = object()
    monkeypatch.setattr(torch, "Event", Event)
    monkeypatch.setattr(head_split.dist, "all_reduce", reduce)
    monkeypatch.setattr(head_split.dist, "get_rank", lambda group: 0)
    trials = head_split.HeadSplitTrials(((), (1, 1)))
    sample = torch.ones(2)
    for _ in range(8):
        assert trials.measure(lambda widths: sample, None, pg, sample.device) is sample
    assert trials.winner == expected
