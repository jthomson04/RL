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

"""Temporary Dynamo MX refit telemetry for the weight-sync investigation."""

from __future__ import annotations

import base64
import gc
import hashlib
import logging
import os
import secrets
import time
import traceback
from typing import Any


def _mx_candidate_version(candidate: Any) -> int | None:
    try:
        return int(candidate.ref.training_step)
    except (AttributeError, TypeError, ValueError):
        return None


def _mx_candidate_role(candidate: Any) -> str:
    return str(getattr(candidate, "role", ""))


def _mx_is_compatible_megatron_trainer(
    candidate: Any,
    *,
    target_tp_rank: int,
    target_tp_size: int,
) -> bool:
    if _mx_candidate_role(candidate) != "trainer":
        return False
    megatron_meta = getattr(candidate, "megatron_meta", None)
    if megatron_meta is None:
        return False
    try:
        source_tp_rank = int(megatron_meta.tp_rank)
        source_tp_size = int(megatron_meta.tp_size)
    except (AttributeError, TypeError, ValueError):
        return False
    return source_tp_rank == target_tp_rank and source_tp_size == target_tp_size


def _mx_is_compatible_megatron_replica(
    candidate: Any,
    *,
    target_tp_rank: int,
) -> bool:
    if _mx_candidate_role(candidate) != "inference_replica":
        return False
    try:
        return int(candidate.worker_rank) == target_tp_rank
    except (AttributeError, TypeError, ValueError):
        return False


def _mx_candidate_source_key(candidate: Any) -> tuple[str, str, str, str]:
    ref = getattr(candidate, "ref", None)
    return (
        str(getattr(ref, "mx_source_id", "")),
        str(getattr(ref, "worker_id", "")),
        str(getattr(candidate, "worker_rank", "")),
        _mx_candidate_role(candidate),
    )


def _mx_candidate_source_key_str(candidate: Any | None) -> str:
    if candidate is None:
        return "none"
    source_id, worker_id, worker_rank, role = _mx_candidate_source_key(candidate)
    return f"{role}:{worker_rank}:{source_id}:{worker_id}"


def _mx_source_selector_key(*, version: int, worker_rank: Any) -> str:
    return "|".join(
        (
            str(version),
            str(worker_rank),
            os.environ.get("POD_NAME", ""),
            os.environ.get("HOSTNAME", ""),
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            os.environ.get("LOCAL_RANK", ""),
            os.environ.get("RANK", ""),
            str(os.getpid()),
        )
    )


def _mx_source_rendezvous_score(candidate: Any, selector_key: str) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(selector_key.encode("utf-8", errors="replace"))
    for part in _mx_candidate_source_key(candidate):
        h.update(b"\0")
        h.update(part.encode("utf-8", errors="replace"))
    return h.digest()


def _choose_from_megatron_pool(
    candidates: list[Any],
    *,
    selector_key: str | None,
) -> Any:
    if selector_key:
        return max(
            candidates,
            key=lambda candidate: (
                _mx_source_rendezvous_score(candidate, selector_key),
                _mx_candidate_source_key(candidate),
            ),
        )
    return secrets.choice(candidates)


def _mx_bytes_to_b64(value: Any) -> str:
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


def _mx_descriptor_to_record(descriptor: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(descriptor, "name")),
        "addr": int(getattr(descriptor, "addr")),
        "size": int(getattr(descriptor, "size")),
        "device_id": int(getattr(descriptor, "device_id")),
        "dtype": str(getattr(descriptor, "dtype")),
    }


def _mx_direct_metadata_from_nixl(nixl: Any) -> dict[str, Any]:
    descriptors = getattr(nixl, "tensor_descriptors", None)
    if descriptors is None:
        descriptors = getattr(nixl, "_tensor_descriptors", [])
    return {
        "nixl_metadata_b64": _mx_bytes_to_b64(
            getattr(nixl, "nixl_metadata", getattr(nixl, "_metadata", b""))
        ),
        "tensors": [_mx_descriptor_to_record(desc) for desc in descriptors],
    }


def _mx_make_inference_replica_candidate(
    *,
    receiver: Any,
    mx_source_id: str,
    model_name: str,
    version: int,
) -> dict[str, Any] | None:
    inner_receiver = getattr(receiver, "_receiver", None)
    nixl = getattr(inner_receiver, "_nixl", None)
    worker_id = getattr(inner_receiver, "_worker_id", "")
    if nixl is None or not worker_id or not mx_source_id:
        return None
    worker_rank = int(getattr(receiver, "worker_rank"))
    return {
        "format": "nemo_rl.mx_source_candidate.v1",
        "ref": {
            "mx_source_id": str(mx_source_id),
            "worker_id": str(worker_id),
            "model_name": str(model_name),
            "worker_rank": worker_rank,
            "training_step": int(version),
        },
        "role": "inference_replica",
        "worker_rank": worker_rank,
        "updated_at": int(time.time() * 1000),
        "direct_metadata": _mx_direct_metadata_from_nixl(nixl),
    }


def _mx_deserialize_source_plan_candidates(
    source_plan: Any,
    *,
    version: int,
    model_name: str,
) -> list[Any]:
    if not isinstance(source_plan, dict):
        return []
    if source_plan.get("format") != "nemo_rl.mx_source_plan.v1":
        return []
    try:
        plan_version = int(source_plan.get("version"))
    except (TypeError, ValueError):
        return []
    if plan_version != int(version):
        return []
    plan_model_name = source_plan.get("model_name")
    if plan_model_name and str(plan_model_name) != str(model_name):
        return []

    from modelexpress.nemo_rl_v2 import (
        MegatronSourceMeta,
        V2SourceCandidate,
    )
    from modelexpress.refit_receiver import SourceRef
    from modelexpress.shape_descriptors import decode_registry

    out: list[Any] = []
    for item in source_plan.get("candidates", []):
        if not isinstance(item, dict):
            continue
        ref = item.get("ref")
        if not isinstance(ref, dict):
            continue
        try:
            candidate_version = int(ref.get("training_step"))
            worker_rank = int(item.get("worker_rank", ref.get("worker_rank")))
        except (TypeError, ValueError):
            continue
        if candidate_version != int(version):
            continue
        candidate_model_name = str(ref.get("model_name", ""))
        if candidate_model_name != str(model_name):
            continue

        registry_blob = item.get("registry_blob") or ""
        registry = decode_registry(registry_blob) if registry_blob else None
        owned_experts_per_layer: dict[int, set[int]] = {}
        for layer, experts in (item.get("owned_experts_per_layer") or {}).items():
            owned_experts_per_layer[int(layer)] = {int(expert) for expert in experts}

        megatron_meta = None
        meta = item.get("megatron_meta")
        if isinstance(meta, dict):
            megatron_meta = MegatronSourceMeta(
                tp_rank=int(meta.get("tp_rank", 0)),
                tp_size=int(meta.get("tp_size", 1)),
                pp_rank=int(meta.get("pp_rank", 0)),
                pp_size=int(meta.get("pp_size", 1)),
                ep_rank=int(meta.get("ep_rank", 0)),
                ep_size=int(meta.get("ep_size", 1)),
            )

        candidate = V2SourceCandidate(
            ref=SourceRef(
                mx_source_id=str(ref.get("mx_source_id", "")),
                worker_id=str(ref.get("worker_id", "")),
                model_name=candidate_model_name,
                worker_rank=worker_rank,
                training_step=candidate_version,
            ),
            role=str(item.get("role", "")),
            worker_rank=worker_rank,
            registry=registry,
            owned_experts_per_layer=owned_experts_per_layer,
            updated_at=int(item.get("updated_at", 0) or 0),
            megatron_meta=megatron_meta,
        )
        setattr(candidate, "_direct_metadata", item.get("direct_metadata") or {})
        out.append(candidate)
    return out


def _mx_source_tensors_from_direct_metadata(direct_metadata: dict[str, Any]) -> list[Any]:
    from modelexpress.types import TensorDescriptor

    tensors: list[Any] = []
    for item in direct_metadata.get("tensors", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        size = int(item.get("size", 0) or 0)
        if name.startswith("__mx_") or size <= 0:
            continue
        tensors.append(
            TensorDescriptor(
                name=name,
                addr=int(item.get("addr", 0)),
                size=size,
                device_id=int(item.get("device_id", 0)),
                dtype=str(item.get("dtype", "")),
            )
        )
    return tensors


def _mx_receive_from_direct_metadata(
    *,
    receiver: Any,
    candidate: Any,
    timeout_seconds: float,
) -> Any:
    direct_metadata = getattr(candidate, "_direct_metadata", None)
    if not isinstance(direct_metadata, dict):
        raise RuntimeError("candidate has no direct MX metadata")
    metadata_b64 = direct_metadata.get("nixl_metadata_b64", "")
    source_metadata = base64.b64decode(metadata_b64)
    source_tensors = _mx_source_tensors_from_direct_metadata(direct_metadata)
    nixl = receiver._receiver._nixl
    nixl.receive_from_source(
        source_metadata=source_metadata,
        source_tensors=source_tensors,
        timeout_seconds=timeout_seconds,
    )
    receiver._receiver._current_step = int(candidate.ref.training_step)
    for td in source_tensors:
        if td.name in nixl._tensors:
            yield td.name, nixl._tensors[td.name]


def _mx_pull_to_from_direct_metadata(
    *,
    receiver: Any,
    candidate: Any,
    requests: list[tuple[str, tuple[int, int] | None, Any]],
    timeout_seconds: float,
) -> tuple[int, int, float]:
    import torch
    from modelexpress.nixl_transfer import SlicedTransferRequest
    from modelexpress.refit_receiver import _DTYPE_MAP

    direct_metadata = getattr(candidate, "_direct_metadata", None)
    if not isinstance(direct_metadata, dict):
        raise RuntimeError("candidate has no direct MX metadata")
    source_metadata = base64.b64decode(direct_metadata.get("nixl_metadata_b64", ""))
    source_tensors = _mx_source_tensors_from_direct_metadata(direct_metadata)
    by_name = {tensor.name: tensor for tensor in source_tensors}

    slice_requests = []
    for name, subslice, dest_view in requests:
        src = by_name.get(name)
        if src is None:
            raise RuntimeError(f"direct pull_to: tensor {name!r} not in source manifest")
        if subslice is None:
            source_offset_bytes = 0
            slice_bytes = int(src.size)
        else:
            lo, hi = subslice
            elem_size = torch.tensor(
                [], dtype=_DTYPE_MAP.get(src.dtype, torch.bfloat16)
            ).element_size()
            source_offset_bytes = int(lo) * elem_size
            slice_bytes = int(hi - lo) * elem_size
        slice_requests.append(
            SlicedTransferRequest(
                name=name,
                source_offset_bytes=source_offset_bytes,
                slice_bytes=slice_bytes,
                dest_view=dest_view,
            )
        )

    transferred, num_slices, elapsed = (
        receiver._receiver._nixl.receive_sliced_from_source(
            source_metadata=source_metadata,
            source_tensors=source_tensors,
            slice_requests=slice_requests,
            timeout_seconds=timeout_seconds,
        )
    )
    receiver._receiver._current_step = int(candidate.ref.training_step)
    if slice_requests and (num_slices <= 0 or transferred <= 0):
        raise RuntimeError(
            "direct pull_to transferred no data "
            f"(slices={num_slices}, bytes={transferred})"
        )
    return transferred, num_slices, elapsed


def _choose_megatron_bulk_source(
    candidates: list[Any],
    *,
    version: int,
    target_tp_rank: int,
    target_tp_size: int,
    selector_key: str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    stats = {
        "total": len(candidates),
        "same_version": 0,
        "missing_version": 0,
        "stale_version": 0,
        "future_version": 0,
        "eligible_trainers": 0,
        "eligible_replicas": 0,
        "selection_pool": "none",
        "selection_pool_size": 0,
    }
    eligible_trainers: list[Any] = []
    eligible_replicas: list[Any] = []
    for candidate in candidates:
        candidate_version = _mx_candidate_version(candidate)
        if candidate_version is None:
            stats["missing_version"] += 1
            continue
        if candidate_version < version:
            stats["stale_version"] += 1
            continue
        if candidate_version > version:
            stats["future_version"] += 1
            continue

        stats["same_version"] += 1
        if _mx_is_compatible_megatron_trainer(
            candidate,
            target_tp_rank=target_tp_rank,
            target_tp_size=target_tp_size,
        ):
            stats["eligible_trainers"] += 1
            eligible_trainers.append(candidate)
        elif _mx_is_compatible_megatron_replica(
            candidate,
            target_tp_rank=target_tp_rank,
        ):
            stats["eligible_replicas"] += 1
            eligible_replicas.append(candidate)

    if eligible_replicas:
        stats["selection_pool"] = "inference_replica"
        stats["selection_pool_size"] = len(eligible_replicas)
        return (
            _choose_from_megatron_pool(
                eligible_replicas,
                selector_key=selector_key
                or _mx_source_selector_key(
                    version=version,
                    worker_rank=target_tp_rank,
                ),
            ),
            stats,
        )
    if eligible_trainers:
        stats["selection_pool"] = "trainer"
        stats["selection_pool_size"] = len(eligible_trainers)
        return (
            _choose_from_megatron_pool(
                eligible_trainers,
                selector_key=selector_key
                or _mx_source_selector_key(
                    version=version,
                    worker_rank=target_tp_rank,
                ),
            ),
            stats,
        )
    else:
        return None, stats


def _mx_nic_pin_min_rate(logger: logging.Logger) -> float | None:
    raw_min = os.environ.get("MX_RDMA_NIC_PIN_MIN_RATE_GBPS")
    if raw_min is None or raw_min.strip() == "":
        return None
    try:
        return float(raw_min)
    except ValueError:
        logger.warning(
            "[weight-sync-debug][mx-dynamo-worker] "
            "MX_RDMA_NIC_PIN_MIN_RATE_GBPS=%r not a float; "
            "falling back to max-rate auto-detect",
            raw_min,
        )
        return None


def _apply_stripe_nic_pin_for_device(device_id: int, logger: logging.Logger) -> bool:
    import modelexpress.ucx_utils as ucx_utils

    list_compute_nics = getattr(ucx_utils, "_list_compute_ib_nics", None)
    if list_compute_nics is None:
        logger.warning(
            "[weight-sync-debug][mx-dynamo-worker] "
            "MX_RDMA_NIC_PIN=stripe fallback unavailable: "
            "modelexpress.ucx_utils._list_compute_ib_nics is missing"
        )
        return False

    compute_nics = list_compute_nics(min_rate_gbps=_mx_nic_pin_min_rate(logger))
    if not compute_nics:
        logger.warning(
            "[weight-sync-debug][mx-dynamo-worker] "
            "MX_RDMA_NIC_PIN=stripe fallback: no compute IB-class NICs found; "
            "skipping pin"
        )
        return False

    pinned = ",".join(f"{name}:1" for name, _numa, _rate, _path in compute_nics)
    prev = os.environ.get("UCX_NET_DEVICES")
    os.environ["UCX_NET_DEVICES"] = pinned
    nic_count = len(compute_nics)
    if nic_count >= 2 and "UCX_MAX_RMA_RAILS" not in os.environ:
        os.environ["UCX_MAX_RMA_RAILS"] = str(nic_count)
    os.environ["MX_RDMA_NIC_PIN"] = "off"
    logger.info(
        "[weight-sync-debug][mx-dynamo-worker] "
        "MX_RDMA_NIC_PIN=stripe fallback: device %d -> UCX_NET_DEVICES=%s "
        "(was: %s), UCX_MAX_RMA_RAILS=%s, MX_RDMA_NIC_PIN=%s",
        device_id,
        pinned,
        prev,
        os.environ.get("UCX_MAX_RMA_RAILS"),
        os.environ.get("MX_RDMA_NIC_PIN"),
    )
    return True


def _patch_modelexpress_stripe_fallback(logger: logging.Logger) -> None:
    import modelexpress.ucx_utils as ucx_utils

    if hasattr(ucx_utils, "_stripe_all_compute_nics"):
        return
    if getattr(ucx_utils, "_nemo_stripe_fallback_patched", False):
        return

    original_apply_nic_pin_for_device = ucx_utils.apply_nic_pin_for_device

    def apply_nic_pin_for_device(device_id: int) -> None:
        mode = os.environ.get("MX_RDMA_NIC_PIN", "").strip().lower()
        if mode not in ("stripe", "all"):
            original_apply_nic_pin_for_device(device_id)
            return

        _apply_stripe_nic_pin_for_device(device_id, logger)

    ucx_utils.apply_nic_pin_for_device = apply_nic_pin_for_device
    ucx_utils._nemo_stripe_fallback_patched = True
    logger.info(
        "[weight-sync-debug][mx-dynamo-worker] "
        "installed temporary ModeExpress stripe NIC fallback"
    )


def _patch_dynamo_extension_stripe_pin(ext: Any, logger: logging.Logger) -> None:
    if getattr(ext, "_nemo_stripe_extension_pin_patched", False):
        return

    original_pin_local_nic = ext._pin_local_nic

    def _pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
        normalized_mode = str(mode or "auto").strip().lower()
        if normalized_mode not in ("stripe", "all"):
            original_pin_local_nic(device_id=device_id, mode=mode)
            return

        if not _apply_stripe_nic_pin_for_device(device_id, logger):
            original_pin_local_nic(device_id=device_id, mode=mode)
        logger.info(
            "[weight-sync-debug][mx-dynamo-worker] "
            "handled DGD extension nic_pin=%s for device %d",
            normalized_mode,
            device_id,
        )

    ext._pin_local_nic = _pin_local_nic
    ext._nemo_stripe_extension_pin_patched = True


def _patch_dynamo_mx_refit() -> None:
    if os.environ.get("DYN_MX_REFIT_ENABLED") != "1":
        return

    import torch
    import dynamo.vllm.mx_refit.extension as ext

    logger = logging.getLogger("dynamo.vllm.mx_refit.extension")
    _patch_modelexpress_stripe_fallback(logger)
    _patch_dynamo_extension_stripe_pin(ext, logger)

    worker_cls = ext.MxRefitWorkerExtension
    if getattr(worker_cls, "_nemo_weight_sync_debug_patched", False):
        return

    def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
        return int(sum(t.numel() * t.element_size() for t in tensors.values()))

    def _cuda_mem() -> str:
        if not torch.cuda.is_available():
            return "cuda_mem=unavailable"
        return (
            f"cuda_alloc_gb={torch.cuda.memory_allocated() / 1e9:.2f} "
            f"cuda_reserved_gb={torch.cuda.memory_reserved() / 1e9:.2f} "
            f"cuda_max_alloc_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}"
        )

    def _log(prefix: str, message: str, *args: Any) -> None:
        logger.info(
            "[weight-sync-debug][mx-dynamo-worker][%s] " + message, prefix, *args
        )

    def _snapshot_call_timer_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
        return dict(stats or {})

    def _format_call_timer_delta(
        stats: dict[str, Any] | None,
        before: dict[str, Any] | None,
        keys: tuple[str, ...],
    ) -> str:
        stats = stats or {}
        before = before or {}
        parts = []
        for key in keys:
            elapsed = float(stats.get(key, 0.0)) - float(before.get(key, 0.0))
            count_key = f"{key}_count"
            count = int(stats.get(count_key, 0)) - int(before.get(count_key, 0))
            parts.append(f"{key}_s={elapsed:.6f} {key}_count={count}")
        return " ".join(parts)

    def _install_mx_call_timers(receiver: Any) -> tuple[dict[str, Any], Any]:
        stats: dict[str, Any] = {}
        restores: list[tuple[Any, str, Any]] = []

        def _wrap(obj: Any, method_name: str, key: str) -> None:
            if obj is None:
                return
            original = getattr(obj, method_name, None)
            if original is None:
                return

            def timed_method(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    stats[key] = float(stats.get(key, 0.0)) + (
                        time.perf_counter() - start
                    )
                    count_key = f"{key}_count"
                    stats[count_key] = int(stats.get(count_key, 0)) + 1

            try:
                setattr(obj, method_name, timed_method)
            except Exception:  # noqa: BLE001
                return
            restores.append((obj, method_name, original))

        inner_receiver = getattr(receiver, "_receiver", None)
        client = getattr(inner_receiver, "_client", None)
        nixl = getattr(inner_receiver, "_nixl", None)
        _wrap(client, "list_sources", "client_list_sources")
        _wrap(client, "get_metadata", "client_get_metadata")
        _wrap(client, "publish_metadata", "client_publish_metadata")
        _wrap(nixl, "register_tensors", "nixl_register_tensors")
        _wrap(nixl, "rebind_tensors", "nixl_rebind_tensors")
        _wrap(nixl, "receive_from_source", "nixl_receive_from_source")
        _wrap(nixl, "receive_sliced_from_source", "nixl_receive_sliced")

        def restore() -> None:
            for obj, method_name, original in restores:
                setattr(obj, method_name, original)

        return stats, restore

    original_dtensor_update = worker_cls._mx_update_weights_via_mx_dtensor

    def update_weights_via_mx(
        self: Any,
        *,
        version: int,
        mx_config: Any = None,
        source_plan: Any = None,
    ) -> bool | dict[str, Any]:
        total_start = time.perf_counter()
        call_timer_stats: dict[str, Any] = {}
        restore_mx_call_timers = lambda: None
        try:
            if not isinstance(mx_config, ext.MxConfig):
                mx_config = ext.MxConfig.from_dict(mx_config or {})

            start = time.perf_counter()
            self._mx_init_receiver(mx_config)
            init_s = time.perf_counter() - start
            call_timer_stats, restore_mx_call_timers = _install_mx_call_timers(
                self._mx_receiver
            )
            self._mx_call_timer_stats = call_timer_stats
            loaded_version = int(getattr(self, "_mx_loaded_version", -1))
            if loaded_version >= int(version):
                _log(
                    "update",
                    "version=%d already_loaded_version=%d; skipping refit",
                    version,
                    loaded_version,
                )
                return True

            start = time.perf_counter()
            discover_call_timers_before = _snapshot_call_timer_stats(call_timer_stats)
            model_name = ext._model_name(self)
            source_plan_present = isinstance(source_plan, dict)
            if source_plan_present:
                candidates = _mx_deserialize_source_plan_candidates(
                    source_plan,
                    version=int(version),
                    model_name=model_name,
                )
            else:
                candidates = self._mx_receiver.discover_v2_sources(
                    model_name=model_name,
                    min_version=int(version),
                    same_rank_only=mx_config.same_rank_only,
                    include_replicas=mx_config.tree_scale_out,
                )
            discover_s = time.perf_counter() - start
            if source_plan_present:
                discover_detail = "source_plan"
            else:
                discover_detail = _format_call_timer_delta(
                    call_timer_stats,
                    discover_call_timers_before,
                    ("client_list_sources", "client_get_metadata"),
                )
            megatron_candidates = sum(c.megatron_meta is not None for c in candidates)
            replica_candidates = sum(
                "inference_replica" in str(getattr(c, "role", "")) for c in candidates
            )
            _log(
                "update",
                (
                    "version=%d init_s=%.6f discover_s=%.6f candidates=%d "
                    "megatron_candidates=%d replica_candidates=%d "
                    "source_plan=%s discover_detail=\"%s\" "
                    "total_pre_dispatch_s=%.6f %s"
                ),
                version,
                init_s,
                discover_s,
                len(candidates),
                megatron_candidates,
                replica_candidates,
                source_plan_present,
                discover_detail,
                time.perf_counter() - total_start,
                _cuda_mem(),
            )
            if not candidates:
                logger.warning(
                    "[mx] no v2 source available for version>=%d on rank %d",
                    version,
                    self._mx_receiver.worker_rank,
                )
                return False

            if megatron_candidates:
                result = self._mx_update_weights_via_mx_megatron(
                    candidates=candidates,
                    version=int(version),
                    mx_config=mx_config,
                )
            else:
                dtensor_start = time.perf_counter()
                result = original_dtensor_update(
                    self,
                    candidates=candidates,
                    version=int(version),
                    mx_config=mx_config,
                )
                _log(
                    "dtensor",
                    "version=%d total_s=%.6f result=%s %s",
                    version,
                    time.perf_counter() - dtensor_start,
                    result,
                    _cuda_mem(),
                )
            _log(
                "update",
                "version=%d total_s=%.6f result=%s %s",
                version,
                time.perf_counter() - total_start,
                result,
                _cuda_mem(),
            )
            if result:
                self._mx_loaded_version = int(version)
            return result if isinstance(result, dict) else bool(result)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[mx] update_weights_via_mx failed on rank=%s: %s\n%s",
                getattr(getattr(self, "_mx_receiver", None), "worker_rank", -1),
                exc,
                traceback.format_exc(),
            )
            return False
        finally:
            restore_mx_call_timers()

    def _mx_pull_megatron_vocab_buffers(
        self: Any,
        *,
        candidates: list[Any],
        ctx: Any,
        mx_config: Any,
    ) -> tuple[int, int, float, float]:
        vocab_buffers = getattr(self, "_mx_megatron_vocab_buffers", {})
        if not vocab_buffers:
            return 0, 0, 0.0, 0.0

        wall_start = time.perf_counter()
        total_bytes = 0
        total_slices = 0
        total_transfer_s = 0.0
        direct_sources = 0
        catalog_sources = 0
        megatron_cands = sorted(
            [c for c in candidates if c.megatron_meta is not None],
            key=lambda c: c.megatron_meta.tp_rank,
        )
        for cand in megatron_cands:
            batch = []
            for name, dest in vocab_buffers.items():
                spec = ctx.receive_specs[name]
                axis = int(spec.shard_axis)
                if axis != 0:
                    raise RuntimeError(
                        f"vocab_parallel tensor {name!r} uses unsupported "
                        f"shard_axis={axis}; expected 0"
                    )
                rows = int(spec.target_shape[axis])
                lo = int(cand.megatron_meta.tp_rank) * rows
                view = dest.narrow(axis, lo, rows)
                if not view.is_contiguous():
                    raise RuntimeError(
                        f"vocab_parallel destination for {name!r} is not contiguous"
                    )
                batch.append((name, None, view))
            if not batch:
                continue
            if isinstance(getattr(cand, "_direct_metadata", None), dict):
                bytes_transferred, slices, transfer_s = _mx_pull_to_from_direct_metadata(
                    receiver=self._mx_receiver,
                    candidate=cand,
                    requests=batch,
                    timeout_seconds=mx_config.timeout_seconds,
                )
                direct_sources += 1
            else:
                bytes_transferred, slices, transfer_s = (
                    self._mx_receiver._receiver.pull_to(
                        cand.ref,
                        batch,
                        timeout_seconds=mx_config.timeout_seconds,
                    )
                )
                catalog_sources += 1
            total_bytes += bytes_transferred
            total_slices += slices
            total_transfer_s += transfer_s
        wall_s = time.perf_counter() - wall_start
        print(
            "[weight-sync-debug][mx-vllm] "
            f"rank={getattr(self._mx_receiver, 'worker_rank', '?')} "
            f"vocab_pull_slices={total_slices} "
            f"vocab_pull_gb={total_bytes / 1e9:.3f} "
            f"vocab_transfer_s={total_transfer_s:.3f} "
            f"vocab_wall_s={wall_s:.3f} "
            f"vocab_direct_sources={direct_sources} "
            f"vocab_catalog_sources={catalog_sources}",
            flush=True,
        )
        return total_bytes, total_slices, total_transfer_s, wall_s

    def _mx_update_weights_via_mx_megatron(
        self: Any,
        *,
        candidates: list[Any],
        version: int,
        mx_config: Any,
    ) -> bool | dict[str, Any]:
        from modelexpress.megatron_translator import run_refit_cycle
        from modelexpress.nemo_rl_v2 import ROLE_MEGATRON_VOCAB_PARALLEL

        total_start = time.perf_counter()
        rank = getattr(getattr(self, "_mx_receiver", None), "worker_rank", -1)
        _log(
            "megatron",
            "version=%d rank=%s start candidates=%d %s",
            version,
            rank,
            len(candidates),
            _cuda_mem(),
        )
        exact_candidates = [
            candidate
            for candidate in candidates
            if _mx_candidate_version(candidate) == int(version)
        ]
        stale_candidates = sum(
            1
            for candidate in candidates
            if (
                _mx_candidate_version(candidate) is not None
                and _mx_candidate_version(candidate) < int(version)
            )
        )
        future_candidates = sum(
            1
            for candidate in candidates
            if (
                _mx_candidate_version(candidate) is not None
                and _mx_candidate_version(candidate) > int(version)
            )
        )
        _log(
            "megatron",
            (
                "version=%d rank=%s exact_candidates=%d stale_candidates=%d "
                "future_candidates=%d"
            ),
            version,
            rank,
            len(exact_candidates),
            stale_candidates,
            future_candidates,
        )
        if not exact_candidates:
            logger.warning(
                "[mx-megatron] no exact-version source available for version %d",
                version,
            )
            return False
        candidates = exact_candidates
        megatron_context_candidates = [
            candidate
            for candidate in candidates
            if getattr(candidate, "megatron_meta", None) is not None
        ]
        if not megatron_context_candidates:
            logger.warning(
                "[mx-megatron] no exact-version Megatron trainer metadata "
                "available for version %d",
                version,
            )
            return False

        start = time.perf_counter()
        if not getattr(self, "_mx_megatron_ctx", None):
            self._mx_megatron_ctx = self._build_megatron_context(
                megatron_context_candidates
            )
            context_built = True
        else:
            context_built = False
        ctx = self._mx_megatron_ctx
        context_s = time.perf_counter() - start
        _log(
            "megatron",
            "version=%d rank=%s context_s=%.6f context_built=%s tensors=%d",
            version,
            rank,
            context_s,
            context_built,
            len(ctx.receive_specs),
        )

        prealloc_call_timers_before = _snapshot_call_timer_stats(
            getattr(self, "_mx_call_timer_stats", {})
        )
        start = time.perf_counter()
        buffers_created = False
        registered_bytes = 0
        if not hasattr(self, "_mx_megatron_buffers"):
            buffers: dict[str, torch.Tensor] = {}
            vocab_buffers: dict[str, torch.Tensor] = {}
            source_tp_size = next(
                (
                    c.megatron_meta.tp_size
                    for c in megatron_context_candidates
                    if c.megatron_meta is not None and c.megatron_meta.tp_size > 0
                ),
                ctx.target_tp_layout.tp_size,
            )
            for spec in ctx.receive_specs.values():
                if spec.role.startswith("expert_"):
                    continue
                shape = list(spec.target_shape)
                target = buffers
                if spec.role == ROLE_MEGATRON_VOCAB_PARALLEL:
                    shape[int(spec.shard_axis)] *= int(source_tp_size)
                    target = vocab_buffers
                target[spec.megatron_name] = torch.empty(
                    shape,
                    dtype=ext._torch_dtype(spec.target_dtype),
                    device=self.device,
                )
            all_buffers = dict(buffers)
            all_buffers.update(vocab_buffers)
            registered_bytes = _tensor_bytes(all_buffers)
            if all_buffers:
                self._mx_receiver._receiver._nixl.register_tensors(all_buffers)
            self._mx_megatron_buffers = buffers
            self._mx_megatron_vocab_buffers = vocab_buffers
            buffers_created = True
            logger.info(
                "[mx-megatron] registered %d per-rank buffers and %d full-vocab buffers",
                len(buffers),
                len(vocab_buffers),
            )
        prealloc_s = time.perf_counter() - start
        prealloc_detail = _format_call_timer_delta(
            getattr(self, "_mx_call_timer_stats", {}),
            prealloc_call_timers_before,
            ("nixl_register_tensors",),
        )
        _log(
            "megatron",
            (
                "version=%d rank=%s prealloc_register_s=%.6f buffers_created=%s "
                "buffers=%d vocab_buffers=%d bytes=%.3fGB "
                "prealloc_detail=\"%s\" %s"
            ),
            version,
            rank,
            prealloc_s,
            buffers_created,
            len(self._mx_megatron_buffers),
            len(self._mx_megatron_vocab_buffers),
            registered_bytes / 1e9,
            prealloc_detail,
            _cuda_mem(),
        )

        matched, selection_stats = _choose_megatron_bulk_source(
            candidates,
            version=int(version),
            target_tp_rank=int(ctx.target_tp_layout.tp_rank),
            target_tp_size=int(ctx.target_tp_layout.tp_size),
        )
        source_tp_size = next(
            (
                c.megatron_meta.tp_size
                for c in megatron_context_candidates
                if c.megatron_meta is not None and c.megatron_meta.tp_size > 0
            ),
            None,
        )
        _log(
            "megatron",
            (
                "version=%d rank=%s selection_total=%d same_version=%d "
                "missing_version=%d stale=%d future=%d eligible_trainers=%d "
                "eligible_replicas=%d selection_pool=%s selection_pool_size=%d "
                "chosen_role=%s chosen_worker_rank=%s chosen_source_id=%s "
                "chosen_worker_id=%s chosen_source_key=%s"
            ),
            version,
            rank,
            selection_stats["total"],
            selection_stats["same_version"],
            selection_stats["missing_version"],
            selection_stats["stale_version"],
            selection_stats["future_version"],
            selection_stats["eligible_trainers"],
            selection_stats["eligible_replicas"],
            selection_stats["selection_pool"],
            selection_stats["selection_pool_size"],
            _mx_candidate_role(matched) if matched is not None else "none",
            getattr(matched, "worker_rank", "none") if matched is not None else "none",
            getattr(getattr(matched, "ref", None), "mx_source_id", "none")
            if matched is not None
            else "none",
            getattr(getattr(matched, "ref", None), "worker_id", "none")
            if matched is not None
            else "none",
            _mx_candidate_source_key_str(matched),
        )
        if matched is None or (
            source_tp_size is not None
            and source_tp_size != ctx.target_tp_layout.tp_size
        ):
            raise RuntimeError(
                "Dynamo Megatron MX refit currently supports matched TP only "
                f"(target_tp={ctx.target_tp_layout.tp_size}, source_tp={source_tp_size})."
            )

        receive_call_timers_before = _snapshot_call_timer_stats(
            getattr(self, "_mx_call_timer_stats", {})
        )
        start = time.perf_counter()
        self._mx_receiver._receiver._nixl.rebind_tensors(self._mx_megatron_buffers)
        rebind_s = time.perf_counter() - start

        start = time.perf_counter()
        received_count = 0
        received_bytes = 0
        if isinstance(getattr(matched, "_direct_metadata", None), dict):
            receive_iter = _mx_receive_from_direct_metadata(
                receiver=self._mx_receiver,
                candidate=matched,
                timeout_seconds=mx_config.timeout_seconds,
            )
            receive_path = "source_plan"
        else:
            receive_iter = self._mx_receiver.receive_from(
                matched,
                timeout_seconds=mx_config.timeout_seconds,
            )
            receive_path = "catalog"
        for _name, _tensor in receive_iter:
            received_count += 1
            received_bytes += int(_tensor.numel() * _tensor.element_size())
        receive_s = time.perf_counter() - start
        receive_detail = _format_call_timer_delta(
            getattr(self, "_mx_call_timer_stats", {}),
            receive_call_timers_before,
            (
                "nixl_rebind_tensors",
                "client_get_metadata",
                "nixl_receive_from_source",
            ),
        )
        receive_gbps = (received_bytes * 8 / receive_s / 1e9) if receive_s > 0 else 0.0
        _log(
            "megatron",
            (
                "version=%d rank=%s rebind_s=%.6f receive_s=%.6f "
                "source_role=%s source_rank=%s source_version=%s "
                "source_id=%s source_worker_id=%s source_key=%s "
                "receive_path=%s received_tensors=%d "
                "received_bytes=%.3fGB "
                "receive_gbps=%.2f receive_detail=\"%s\" %s"
            ),
            version,
            rank,
            rebind_s,
            receive_s,
            _mx_candidate_role(matched),
            matched.worker_rank,
            _mx_candidate_version(matched),
            matched.ref.mx_source_id,
            matched.ref.worker_id,
            _mx_candidate_source_key_str(matched),
            receive_path,
            received_count,
            received_bytes / 1e9,
            receive_gbps,
            receive_detail,
            _cuda_mem(),
        )
        if received_count <= 0 or received_bytes <= 0:
            raise RuntimeError(
                "MX Megatron receive transferred no data "
                f"(path={receive_path}, tensors={received_count}, bytes={received_bytes})"
            )

        vocab_call_timers_before = _snapshot_call_timer_stats(
            getattr(self, "_mx_call_timer_stats", {})
        )
        start = time.perf_counter()
        self._mx_pull_megatron_vocab_buffers(
            candidates=megatron_context_candidates,
            ctx=ctx,
            mx_config=mx_config,
        )
        vocab_s = time.perf_counter() - start
        vocab_detail = _format_call_timer_delta(
            getattr(self, "_mx_call_timer_stats", {}),
            vocab_call_timers_before,
            ("client_get_metadata", "nixl_receive_sliced"),
        )

        def _noop_pull(_src: Any, _dest: torch.Tensor) -> None:
            return

        pre_assembled_buffers = dict(self._mx_megatron_buffers)
        pre_assembled_buffers.update(self._mx_megatron_vocab_buffers)
        start = time.perf_counter()
        weights = list(
            run_refit_cycle(
                self._mx_receiver,
                candidates=megatron_context_candidates,
                context=ctx,
                pull=_noop_pull,
                device=self.device,
                pre_assembled_buffers=pre_assembled_buffers,
            )
        )
        translate_s = time.perf_counter() - start
        translated_bytes = sum(t.numel() * t.element_size() for _name, t in weights)
        _log(
            "megatron",
            (
                "version=%d rank=%s vocab_s=%.6f vocab_detail=\"%s\" "
                "translate_s=%.6f "
                "translated_tensors=%d translated_bytes=%.3fGB %s"
            ),
            version,
            rank,
            vocab_s,
            vocab_detail,
            translate_s,
            len(weights),
            translated_bytes / 1e9,
            _cuda_mem(),
        )
        if not weights:
            logger.warning(
                "[mx-megatron] no translated tensors for version %d", version
            )
            return False

        start = time.perf_counter()
        self._mx_load_weights(weights)
        load_s = time.perf_counter() - start

        start = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        sync_s = time.perf_counter() - start

        start = time.perf_counter()
        self._mx_maybe_process_fp8_kv_cache()
        fp8_s = time.perf_counter() - start

        publish_self_call_timers_before = _snapshot_call_timer_stats(
            getattr(self, "_mx_call_timer_stats", {})
        )
        start = time.perf_counter()
        published_source_candidate = None
        if mx_config.tree_scale_out:
            published_source_id = self._mx_receiver.publish_self_as_source(
                version=int(version),
                model_name=ext._model_name(self),
            )
            if published_source_id:
                published_source_candidate = _mx_make_inference_replica_candidate(
                    receiver=self._mx_receiver,
                    mx_source_id=published_source_id,
                    model_name=ext._model_name(self),
                    version=int(version),
                )
        publish_self_s = time.perf_counter() - start
        publish_self_detail = _format_call_timer_delta(
            getattr(self, "_mx_call_timer_stats", {}),
            publish_self_call_timers_before,
            ("client_publish_metadata",),
        )

        start = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        gc_s = time.perf_counter() - start

        _log(
            "megatron",
            (
                "version=%d rank=%s load_s=%.6f sync_s=%.6f fp8_s=%.6f "
                "publish_self_s=%.6f publish_self_detail=\"%s\" "
                "published_source_candidate=%s gc_empty_cache_s=%.6f "
                "total_s=%.6f %s"
            ),
            version,
            rank,
            load_s,
            sync_s,
            fp8_s,
            publish_self_s,
            publish_self_detail,
            published_source_candidate is not None,
            gc_s,
            time.perf_counter() - total_start,
            _cuda_mem(),
        )
        self._mx_loaded_version = int(version)
        return {
            "status": "ok",
            "source_candidates": [published_source_candidate]
            if published_source_candidate is not None
            else [],
        }

    worker_cls.update_weights_via_mx = update_weights_via_mx
    worker_cls._mx_pull_megatron_vocab_buffers = _mx_pull_megatron_vocab_buffers
    worker_cls._mx_update_weights_via_mx_megatron = _mx_update_weights_via_mx_megatron
    worker_cls._nemo_weight_sync_debug_patched = True
    logger.info(
        "[weight-sync-debug][mx-dynamo-worker] installed temporary MX refit patch"
    )


def _patch_dynamo_mx_handlers() -> None:
    try:
        import dynamo.vllm.handlers as handlers
    except Exception:  # noqa: BLE001
        return

    logger = logging.getLogger("weight_sync_debug.mx_dynamo_handler")

    def _worker_result_ok(result: Any) -> bool:
        if isinstance(result, dict):
            return result.get("status") == "ok"
        return bool(result)

    async def update_weights_via_mx(self: Any, request: Any = None) -> Any:
        if request is None:
            yield {
                "status": "error",
                "message": "request body required: {'version': int, 'mx_config': {...}}",
            }
            return
        if not isinstance(request, dict):
            yield {"status": "error", "message": "request body must be a JSON object"}
            return

        version = request.get("version")
        if version is None:
            yield {"status": "error", "message": "'version' (int) is required"}
            return

        mx_config = request.get("mx_config") or {}
        rpc_kwargs = {"version": int(version), "mx_config": mx_config}
        if "source_plan" in request:
            rpc_kwargs["source_plan"] = request["source_plan"]

        try:
            results = await self.engine_client.collective_rpc(
                "update_weights_via_mx",
                kwargs=rpc_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[mx] collective_rpc failed: %s", exc)
            yield {
                "status": "error",
                "message": f"collective_rpc failed: {exc}",
            }
            return

        source_candidates: list[dict[str, Any]] = []
        for result in results or []:
            if isinstance(result, dict):
                nested = result.get("source_candidates") or []
                if isinstance(nested, list):
                    source_candidates.extend(
                        item for item in nested if isinstance(item, dict)
                    )

        all_ok = bool(results) and all(_worker_result_ok(result) for result in results)
        yield {
            "status": "ok" if all_ok else "error",
            "version": int(version),
            "workers_ok": sum(1 for result in (results or []) if _worker_result_ok(result)),
            "workers_total": len(results or []),
            "source_candidates": source_candidates,
        }

    patched = False
    for cls_name in ("BaseWorkerHandler", "DecodeWorkerHandler", "PrefillWorkerHandler"):
        cls = getattr(handlers, cls_name, None)
        if cls is None or getattr(cls, "_nemo_weight_sync_debug_handler_patched", False):
            continue
        setattr(cls, "update_weights_via_mx", update_weights_via_mx)
        setattr(cls, "_nemo_weight_sync_debug_handler_patched", True)
        patched = True
    if patched:
        logger.info(
            "[weight-sync-debug][mx-dynamo-handler] installed source-plan handler patch"
        )


try:
    _patch_dynamo_mx_handlers()
    _patch_dynamo_mx_refit()
except Exception:  # noqa: BLE001
    logging.getLogger(__name__).exception(
        "[weight-sync-debug][mx-dynamo-worker] failed to install temporary MX refit patch"
    )
