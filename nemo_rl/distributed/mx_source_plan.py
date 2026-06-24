# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JSON-safe ModelExpress source-plan helpers.

The MX steady-state refit path should not have to rediscover the same source
metadata from every inference worker. These helpers serialize exactly the
metadata a worker needs to pull from a source directly:

* v2 source identity and exact training-step version;
* the encoded shape registry for trainer sources;
* NIXL agent metadata plus tensor descriptors for direct RDMA receive.

The module intentionally avoids importing ModeExpress at import time so host
unit tests can import NeMo-RL without the runtime MX package installed.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterable
from typing import Any

SOURCE_CANDIDATE_FORMAT = "nemo_rl.mx_source_candidate.v1"
SOURCE_PLAN_FORMAT = "nemo_rl.mx_source_plan.v1"


def _bytes_to_b64(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, bytes):
        raw = value
    elif isinstance(value, bytearray):
        raw = bytes(value)
    elif isinstance(value, memoryview):
        raw = value.tobytes()
    else:
        raw = bytes(value)
    return base64.b64encode(raw).decode("ascii")


def _tensor_descriptor_to_record(descriptor: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(descriptor, "name")),
        "addr": int(getattr(descriptor, "addr")),
        "size": int(getattr(descriptor, "size")),
        "device_id": int(getattr(descriptor, "device_id")),
        "dtype": str(getattr(descriptor, "dtype")),
    }


def _direct_metadata_from_nixl(nixl: Any) -> dict[str, Any]:
    descriptors = getattr(nixl, "tensor_descriptors", None)
    if descriptors is None:
        descriptors = getattr(nixl, "_tensor_descriptors", [])
    return {
        "nixl_metadata_b64": _bytes_to_b64(
            getattr(nixl, "nixl_metadata", getattr(nixl, "_metadata", b""))
        ),
        "tensors": [
            _tensor_descriptor_to_record(descriptor) for descriptor in descriptors
        ],
    }


def make_source_candidate(
    *,
    mx_source_id: str,
    worker_id: str,
    model_name: str,
    worker_rank: int,
    training_step: int,
    role: str,
    direct_metadata: dict[str, Any],
    registry_blob: str | None = None,
    megatron_meta: dict[str, int] | None = None,
    owned_experts_per_layer: dict[int, Iterable[int]] | None = None,
    updated_at: int | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe source candidate."""
    candidate: dict[str, Any] = {
        "format": SOURCE_CANDIDATE_FORMAT,
        "ref": {
            "mx_source_id": str(mx_source_id),
            "worker_id": str(worker_id),
            "model_name": str(model_name),
            "worker_rank": int(worker_rank),
            "training_step": int(training_step),
        },
        "role": str(role),
        "worker_rank": int(worker_rank),
        "updated_at": int(updated_at if updated_at is not None else time.time() * 1000),
        "direct_metadata": direct_metadata,
    }
    if registry_blob:
        candidate["registry_blob"] = str(registry_blob)
    if megatron_meta:
        candidate["megatron_meta"] = {k: int(v) for k, v in megatron_meta.items()}
    if owned_experts_per_layer:
        candidate["owned_experts_per_layer"] = {
            str(layer): [int(expert) for expert in experts]
            for layer, experts in owned_experts_per_layer.items()
        }
    return candidate


def make_source_candidate_from_v2_publisher(
    publisher: Any,
    *,
    model_name: str,
    training_step: int,
    role: str = "trainer",
    registry_blob: str | None = None,
    megatron_meta: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """Serialize a live ``MxV2TrainingPublisher`` without catalog lookup."""
    mx_source_id = getattr(publisher, "mx_source_id", None)
    worker_id = getattr(publisher, "worker_id", None)
    if not mx_source_id or not worker_id:
        return None

    inner = getattr(publisher, "_publisher", None)
    nixl = getattr(inner, "_nixl", None)
    if nixl is None:
        return None

    return make_source_candidate(
        mx_source_id=mx_source_id,
        worker_id=worker_id,
        model_name=model_name,
        worker_rank=int(getattr(publisher, "worker_rank")),
        training_step=int(training_step),
        role=role,
        registry_blob=registry_blob,
        megatron_meta=megatron_meta,
        direct_metadata=_direct_metadata_from_nixl(nixl),
    )


def _source_candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    ref = candidate.get("ref") if isinstance(candidate, dict) else None
    if not isinstance(ref, dict):
        return ("", "")
    return (str(ref.get("mx_source_id", "")), str(ref.get("worker_id", "")))


def dedupe_source_candidates(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate candidates by MX source id and worker id, preserving order."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = _source_candidate_key(candidate)
        if key == ("", "") or key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def source_candidates_from_results(results: Any) -> list[dict[str, Any]]:
    """Flatten source candidates returned from Ray or Dynamo result objects."""
    candidates: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("format") == SOURCE_CANDIDATE_FORMAT and isinstance(
                value.get("ref"), dict
            ):
                candidates.append(value)
                return
            nested = value.get("source_candidates")
            if nested is not None:
                visit(nested)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(results)
    return dedupe_source_candidates(candidates)


def make_source_plan(
    *,
    version: int,
    candidates: Iterable[dict[str, Any]],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build an exact-version source plan for one refit version."""
    exact_candidates: list[dict[str, Any]] = []
    for candidate in dedupe_source_candidates(candidates):
        ref = candidate.get("ref")
        if not isinstance(ref, dict):
            continue
        try:
            candidate_version = int(ref.get("training_step"))
        except (TypeError, ValueError):
            continue
        if candidate_version == int(version):
            exact_candidates.append(candidate)
    plan: dict[str, Any] = {
        "format": SOURCE_PLAN_FORMAT,
        "version": int(version),
        "candidates": exact_candidates,
    }
    if model_name:
        plan["model_name"] = model_name
    return plan
