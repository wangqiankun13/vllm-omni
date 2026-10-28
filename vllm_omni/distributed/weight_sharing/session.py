# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-local bootstrap for a rank-paired weight-sharing session.

The startup/lifecycle layer owns the fields that must agree across processes:
group id, generation, session id, peer ids, the per-rank physical GPU UUID, and
the Unix endpoint/token.  :class:`SharedWeightGroupConfig` is the worker-local
view of that agreement.  It never opens sockets or CUDA resources itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from .contracts import Identity, SessionContext

DEFAULT_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class SharedWeightGroupConfig:
    """Static agreement for one rank of a sharing group.

    ``device_index`` is intentionally absent: it is a per-process, torch-visible
    ordinal supplied by the worker when it assembles the :class:`SessionContext`.
    """

    group_id: str
    generation: int
    session_id: str
    provider_id: str
    consumer_id: str
    role: Literal["provider", "consumer"]
    rank: int
    tp_size: int
    device_uuid: str
    endpoint: str
    token: bytes
    model_id: str
    checkpoint_revision: str
    adapter: str = "hunyuan_image3"
    layout: str = "hunyuan_image3_backbone_v1"


def build_identity(group: SharedWeightGroupConfig) -> Identity:
    return Identity(
        group_id=group.group_id,
        generation=group.generation,
        session_id=group.session_id,
        provider_id=group.provider_id,
        consumer_id=group.consumer_id,
        rank=group.rank,
        tp_size=group.tp_size,
        device_uuid=group.device_uuid,
        protocol_version=DEFAULT_PROTOCOL_VERSION,
    )


def build_session_context(group: SharedWeightGroupConfig, device_index: int) -> SessionContext:
    """Assemble the backend provider/consumer context for a torch-visible ordinal."""
    return SessionContext(
        identity=build_identity(group),
        device_index=device_index,
        endpoint=group.endpoint,
        token=group.token,
    )


def resolve_device_uuid(device_index: int) -> str:
    """Resolve the canonical physical GPU UUID for a torch-visible ordinal.

    Mirrors the canonicalization used by the CUDA IPC codec so orchestrators can
    fill ``SharedWeightGroupConfig.device_uuid`` without duplicating the logic.
    """
    from vllm.utils.platform_utils import cuda_get_device_properties

    (uuid,) = cuda_get_device_properties(device_index, ("uuid",), init_cuda=True)
    value = str(uuid).removeprefix("GPU-")
    return f"GPU-{UUID(value)}"
