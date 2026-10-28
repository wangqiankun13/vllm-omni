# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real CUDA IPC integration test for the HunyuanImage 3.0 adapter.

Mirrors the backend's spawned-process pattern: a provider process publishes a
synthetic backbone and a consumer process reconstructs it on ``meta``, imports
the provider storage through the adapter and attaches views.  Values are checked
against a deterministic CPU reference so the run proves the adapter carried the
provider's finalized weights without a second copy.
"""

import gc
import importlib

import pytest
import torch
from torch import nn

from tests.distributed.weight_sharing.cuda_helpers import receive, require_cuda, spawn_workers
from tests.distributed.weight_sharing.helpers import load_weight_sharing
from tests.helpers.mark import hardware_marks

require_cuda(1)
pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

DIM = 8


def _tensor(shape, device, dtype=torch.float32):
    if device == "meta":
        return torch.empty(*shape, device="meta", dtype=dtype)
    # Generate on CPU (deterministic across processes) and move to the target.
    return torch.randn(*shape, dtype=dtype).to(device)


def _make_backbone(device):
    torch.manual_seed(0)

    def param(shape, dtype=torch.float32):
        return nn.Parameter(_tensor(shape, device, dtype))

    def linear(out, inp):
        module = nn.Module()
        module.register_parameter("weight", param((out, inp)))
        module.register_parameter("bias", param((out,)))
        return module

    def norm(dim):
        module = nn.Module()
        module.register_parameter("weight", param((dim,)))
        return module

    backbone = nn.Module()

    embed = nn.Module()
    embed.register_parameter("weight", param((16, DIM)))
    backbone.embed_tokens = embed

    layer = nn.Module()
    layer.input_layernorm = norm(DIM)
    layer.post_attention_layernorm = norm(DIM)
    attn = nn.Module()
    attn.qkv_proj = linear(3 * DIM, DIM)
    attn.o_proj = linear(DIM, DIM)
    layer.self_attn = attn
    mlp = nn.Module()
    gate = nn.Module()
    gate.register_parameter("weight", param((DIM, DIM), torch.float32))
    mlp.gate = gate
    experts = nn.Module()
    routed = nn.Module()
    routed.register_parameter("w13_weight", param((2 * DIM, DIM)))
    routed.register_parameter("w2_weight", param((DIM, DIM)))
    experts.routed_experts = routed
    mlp.experts = experts
    layer.mlp = mlp
    backbone.layers = nn.ModuleList([layer])

    backbone.norm = norm(DIM)
    return backbone


class _Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = _make_backbone(device)


def _adapter_runtime(endpoint):
    ws = load_weight_sharing()
    session = importlib.import_module(f"{ws.__name__}.session")
    adapter = importlib.import_module(f"{ws.__name__}.adapters.hunyuan_image3")
    codec = importlib.import_module(f"{ws.__name__}.torch_cuda_ipc_codec")
    return ws, session, adapter, codec, codec._device_uuid(0)


def _make_group(session, *, role, endpoint, device_uuid):
    return session.SharedWeightGroupConfig(
        group_id="group",
        generation=1,
        session_id="session",
        provider_id="ar-0",
        consumer_id="dit-0",
        role=role,
        rank=0,
        tp_size=1,
        device_uuid=device_uuid,
        endpoint=endpoint,
        token=b"0123456789abcdef",
        model_id="synthetic",
        checkpoint_revision="revision",
    )


def _provider(endpoint, control):
    ws, session, adapter, _codec, device_uuid = _adapter_runtime(endpoint)
    model = _Model("cuda")
    group = _make_group(session, role="provider", endpoint=endpoint, device_uuid=device_uuid)
    backend = ws.TorchCudaIpcBackend()
    export = adapter.publish_backbone(backend, model, group=group, device_index=0)
    control.send("ready")
    receive(control, "release")
    assert backend.release_export(export).status == "CLOSED"
    control.send("closed")
    receive(control, "next")


def _consumer(endpoint, control):
    ws, session, adapter, _codec, device_uuid = _adapter_runtime(endpoint)
    receive(control, "import")
    model = _Model("meta")
    reference = _make_backbone("cpu")
    group = _make_group(session, role="consumer", endpoint=endpoint, device_uuid=device_uuid)
    backend = ws.TorchCudaIpcBackend()
    imported = adapter.open_backbone_import(backend, model, group=group, device_index=0)

    for name, expected in reference.named_parameters():
        if not adapter.is_shared_param(name):
            continue
        param = model.model.get_parameter(name)
        assert param.device.type == "cuda", name
        assert torch.equal(param.detach().cpu(), expected), name

    # Drop the loop variables, which otherwise keep the last attached CUDA
    # Parameter (and its storage) alive through ``close_import``.
    del param, expected

    # The shared router observes the float32 fusion contract.
    assert model.model.get_parameter("layers.0.mlp.gate.weight").dtype == torch.float32

    # Drop external Parameter references before closing so storage leases expire
    # and the release handshake completes (CLOSED, not HELD).
    del model
    del reference
    gc.collect()
    assert backend.close_import(imported).status == "CLOSED"
    control.send("closed")


def test_two_spawned_processes_share_adapter_weights(tmp_path, ipc_allocator):
    endpoint = str(tmp_path / "weights.sock")
    with spawn_workers((_provider, (endpoint,)), (_consumer, (endpoint,))) as workers:
        (_, provider), (_, consumer) = workers
        receive(provider, "ready")
        consumer.send("import")
        receive(consumer, "closed")
        provider.send("release")
        receive(provider, "closed")
        provider.send("next")
