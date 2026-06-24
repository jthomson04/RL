# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import gc
import hashlib
import os
import secrets
import time
import traceback
from typing import Any

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


def fix_gpt_oss_export_transpose(key: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply GPT-OSS down_proj transpose fix to the weight.

    This is a workaround for the issue that the down_proj layout is not the same across different frameworks.
        - HF needs [in, out] layout.
        - Megatron needs [in, out] layout.
        - vLLM needs [out, in] layout.
    See https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3271 for more details.
    """
    if key.endswith("mlp.experts.down_proj"):
        weight = weight.transpose(-2, -1).contiguous()
    return weight


def _target_tp(worker: Any) -> tuple[int, int]:
    parallel_config = getattr(worker, "parallel_config", None)
    if parallel_config is None:
        vllm_config = getattr(worker.model_runner, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    if tp_size <= 1 and torch.distributed.is_initialized():
        tp_size = int(torch.distributed.get_world_size())
    if torch.distributed.is_initialized():
        tp_rank = int(torch.distributed.get_rank() % tp_size)
    else:
        tp_rank = 0
    return tp_size, tp_rank


def _mx_candidate_version(candidate: Any) -> int | None:
    try:
        return int(candidate.ref.training_step)
    except (AttributeError, TypeError, ValueError):
        return None


def _mx_candidate_role(candidate: Any) -> str:
    return str(getattr(candidate, "role", ""))


def _mx_is_inference_replica(candidate: Any) -> bool:
    return _mx_candidate_role(candidate) == "inference_replica"


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
    if not _mx_is_inference_replica(candidate):
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


def _param_for_loaded_weight(
    name: str,
    params: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    candidates = [name]
    if name.startswith("backbone."):
        candidates.append(f"model.{name[len('backbone.') :]}")
    for candidate in candidates:
        param = params.get(candidate)
        if param is not None:
            return param

    for candidate in candidates:
        for shard_name in ("q_proj", "k_proj", "v_proj"):
            if shard_name not in candidate:
                continue
            mapped_name = candidate.replace(shard_name, "qkv_proj")
            param = params.get(mapped_name)
            if param is not None:
                return param
    return None


def _maybe_copy_tp_local_weight(
    *,
    name: str,
    weight: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> bool:
    """Copy exact TP-local linear shards directly.

    Matched-TP Megatron MX receives local shards. vLLM's standard loaders
    usually expect checkpoint-global tensors and slice again, which is wrong
    for exact-shaped row-parallel and fused local layouts such as Nemotron-H
    Mamba ``conv1d``/``in_proj``.
    """
    param = _param_for_loaded_weight(name, params)
    if param is None or tuple(param.shape) != tuple(weight.shape):
        return False
    if ".experts." in name:
        return False

    is_linear_shard = (
        getattr(param, "input_dim", None) is not None
        or getattr(param, "output_dim", None) is not None
    )
    if not is_linear_shard:
        return False

    with torch.no_grad():
        param.copy_(weight, non_blocking=True)
    return True


def _maybe_expand_tp_local_weight(
    *,
    name: str,
    weight: torch.Tensor,
    params: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Wrap a local TP shard in a checkpoint-global tensor for vLLM loaders."""
    if tp_size <= 1 or weight.ndim == 0:
        return weight

    param = _param_for_loaded_weight(name, params)
    if param is None:
        return weight

    is_sharded_weight = bool(getattr(param, "is_sharded_weight", False))
    use_bitsandbytes_4bit = bool(getattr(param, "use_bitsandbytes_4bit", False))
    if is_sharded_weight or use_bitsandbytes_4bit:
        return weight

    dim = getattr(param, "output_dim", None)
    if dim is None:
        dim = getattr(param, "input_dim", None)
    if dim is None:
        return weight
    dim = int(dim)
    if dim < 0:
        dim += weight.ndim
    if dim < 0 or dim >= weight.ndim:
        return weight

    local_extent = int(weight.shape[dim])
    if dim < param.ndim and local_extent > int(param.shape[dim]):
        return weight

    expanded_shape = list(weight.shape)
    expanded_shape[dim] = local_extent * tp_size
    expanded = torch.empty(
        expanded_shape,
        dtype=weight.dtype,
        device=weight.device,
    )
    expanded.narrow(dim, tp_rank * local_extent, local_extent).copy_(
        weight,
        non_blocking=True,
    )
    return expanded


class VllmInternalWorkerExtension:
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                target_device,
            )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: torch.nn.Module,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

    def _load_weights(self, weights):
        """Load weights with GptOss transpose fix, FP8, and draft-weight support.

        Applies GPT-OSS down_proj transpose if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "GptOssForCausalLM"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weight = fix_gpt_oss_export_transpose(key, weight)
                weights[idx] = (key, weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        tp_size, tp_rank = _target_tp(self)
        if tp_size > 1:
            params = dict(self.model_runner.model.named_parameters())
            adapted_policy_weights = []
            for key, weight in policy_weights:
                if _maybe_copy_tp_local_weight(
                    name=key,
                    weight=weight,
                    params=params,
                ):
                    continue
                adapted_policy_weights.append(
                    (
                        key,
                        _maybe_expand_tp_local_weight(
                            name=key,
                            weight=weight,
                            params=params,
                            tp_size=tp_size,
                            tp_rank=tp_rank,
                        ),
                    )
                )
            policy_weights = adapted_policy_weights
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    # means the update is done
                    from vllm.config import set_current_vllm_config
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

                    with set_current_vllm_config(self.model_runner.vllm_config):
                        process_weights_after_loading(
                            self.model_runner.model, self.model_config, self.device
                        )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weight = None
                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)

                    # Get the weight from the buffer
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weight = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    # apply gpt-oss transpose fix
                    if (
                        "GptOssForCausalLM"
                        in self.model_runner.vllm_config.model_config.architectures
                    ):
                        weight = fix_gpt_oss_export_transpose(key, weight)
                    weights.append((key, weight))

                    # Move offset to the next weight
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weight, weights, buffer
                weight = None
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_mx")
    def update_weights_via_mx(self, *, version: int, mx_config: Any) -> bool:
        """Receive weights via NIXL RDMA from MX server (v2 path).

        Lazy-creates an :class:`MxV2RefitReceiver`, registers our model's
        live parameters once, then for each version: discover same-rank
        source, RDMA receive, slice into per-name views via the trainer's
        published shape registry, hand off to ``_load_weights``, and
        (optionally) republish self as an inference replica for tree
        fan-out.

        Megatron-MX path: if the discovered source's v2 metadata carries
        ``publisher_kind == "megatron"`` (set by Megatron-Core trainers
        per ``modelexpress.megatron_translator.SIDECAR_*`` keys), route
        through :func:`_update_weights_via_mx_megatron` instead. The
        Megatron path uses the receiver-side slice planner +
        Bridge-shaped translator (``modelexpress.megatron_translator``)
        to assemble per-rank shards into HF tensors via the vendored
        QKV un-interleave + gated-MLP split helpers. The translator
        does not depend on Megatron-Bridge being installed in the
        worker image.

        Returns ``True`` on successful refit.
        """
        try:
            assert self.state_dict_info is not None, (
                "state_dict_info not prepared; call prepare_refit_info() first"
            )

            # First-cycle Megatron check: peek at any cached candidates we
            # may have, otherwise discover for the megatron-mode flag and
            # cache. The Megatron path has its own discover/plan loop, so
            # we only do enough discover here to detect the publisher kind.
            if not hasattr(self, "_mx_megatron_mode"):
                self._mx_megatron_mode = None  # None = unknown, True/False = latched

            # ---- Lazy-init receiver and register receive buffers (once) ----
            if not hasattr(self, "_mx_receiver") or self._mx_receiver is None:
                from nemo_rl.distributed.mx_helpers import build_v2_receiver

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
                self._mx_receiver = build_v2_receiver(
                    rank=rank,
                    device_id=self.device.index,
                    mx_config=mx_config,
                )

                # Build receive buffer dict from current model parameters.
                # The trainer publishes local DTensor shards; on the inference
                # side we want vLLM's already-allocated parameters as the
                # destination buffers (no extra copy). vLLM stores them on
                # ``self.model_runner.model``; iterate named_parameters to get
                # them.
                receive_buffers = {
                    name: p.data
                    for name, p in self.model_runner.model.named_parameters()
                    if p.is_cuda
                }
                self._mx_receiver.initialize(model_tensors=receive_buffers)
                self._mx_recv_buffers = receive_buffers

            loaded_version = int(getattr(self, "_mx_loaded_version", -1))
            if loaded_version >= int(version):
                print(
                    "[weight-sync-debug][mx-vllm] "
                    f"rank={self._mx_receiver.worker_rank} version={version} "
                    f"already_loaded_version={loaded_version}; skipping refit",
                    flush=True,
                )
                return True

            # ---- Discover, pick, and pull ----
            candidates = self._mx_receiver.discover_v2_sources(
                model_name=self.model_config.model
                if hasattr(self.model_config, "model")
                else getattr(
                    self.model_runner.vllm_config.model_config, "model", "unknown"
                ),
                min_version=int(version),
                same_rank_only=mx_config.same_rank_only,
                include_replicas=mx_config.tree_scale_out,
            )
            if not candidates:
                print(
                    f"[mx] no v2 source available for version>={version} on rank "
                    f"{self._mx_receiver.worker_rank}"
                )
                return False

            # Latch the receiver mode on the first non-empty discovery.
            if self._mx_megatron_mode is None:
                self._mx_megatron_mode = any(
                    c.megatron_meta is not None for c in candidates
                )
                if self._mx_megatron_mode:
                    print(
                        f"[mx] rank={self._mx_receiver.worker_rank} latched "
                        f"Megatron-MX receiver mode (sources advertise "
                        f"publisher_kind=megatron)"
                    )

            if self._mx_megatron_mode:
                return self._update_weights_via_mx_megatron(
                    candidates=candidates,
                    version=int(version),
                    mx_config=mx_config,
                )

            chosen = self._mx_receiver.pick_best_source(candidates)
            if chosen is None:
                print(
                    f"[mx] no candidate covers required experts on rank "
                    f"{self._mx_receiver.worker_rank}"
                )
                return False
            print(
                f"[mx] rank={self._mx_receiver.worker_rank} chosen source "
                f"role={chosen.role} src_rank={chosen.worker_rank} "
                f"version={chosen.ref.training_step}"
            )

            # Drain RDMA receive into our pre-registered buffers.
            for _name, _tensor in self._mx_receiver.receive_from(
                chosen, timeout_seconds=mx_config.timeout_seconds
            ):
                # The yielded tensor is a view into the same buffer we
                # registered. vLLM's model parameters now hold the new bytes;
                # we still call _load_weights below for FP8 / GptOss /
                # draft-weight handling.
                pass

            # Build (name, weight) pairs for _load_weights from buffers.
            weights = []
            for name, buf in self._mx_recv_buffers.items():
                w = buf
                # apply gpt-oss transpose fix on the way in
                if (
                    "GptOssForCausalLM"
                    in self.model_runner.vllm_config.model_config.architectures
                ):
                    w = fix_gpt_oss_export_transpose(name, w)
                weights.append((name, w))

            self._load_weights(weights)
            torch.cuda.current_stream().synchronize()

            # FP8 KV cache hook reuse
            self._maybe_process_fp8_kv_cache()

            # ---- Tree fan-out: republish self as inference_replica ----
            if mx_config.tree_scale_out:
                self._mx_receiver.publish_self_as_source(
                    version=int(version),
                    model_name=self.model_config.model
                    if hasattr(self.model_config, "model")
                    else getattr(
                        self.model_runner.vllm_config.model_config,
                        "model",
                        "unknown",
                    ),
                )

            gc.collect()
            torch.cuda.empty_cache()
            self._mx_loaded_version = int(version)
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_mx: {e}\n"
                f"{traceback.format_exc()}"
            )
            return False

    def _mx_pull_megatron_vocab_buffers(
        self,
        *,
        candidates: list,
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
            if batch:
                bytes_transferred, slices, transfer_s = (
                    self._mx_receiver._receiver.pull_to(
                        cand.ref,
                        batch,
                        timeout_seconds=mx_config.timeout_seconds,
                    )
                )
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
            f"vocab_wall_s={wall_s:.3f}",
            flush=True,
        )
        return total_bytes, total_slices, total_transfer_s, wall_s

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_via_mx_megatron"
    )
    def _update_weights_via_mx_megatron(
        self,
        *,
        candidates: list,
        version: int,
        mx_config: Any,
    ) -> bool:
        """Megatron-MX path of :meth:`update_weights_via_mx`.

        Routes through ``modelexpress.megatron_translator``'s slice
        planner + assembly pipeline. The trainer publishes per-rank
        Megatron-native shards (no allgather); we discover the slice
        plan, pull each rank's contribution into a pre-allocated global
        tensor, and apply role-aware translation (QKV un-interleave,
        gated-MLP split, name remap) using the vendored helpers — no
        Megatron-Bridge import required in the worker image.

        See ``temp/NemoRL_Megatron_MX_Design.md`` §6 + §9b and
        ``temp/NemoRL_Megatron_MX_Phase_C_Handoff.md``.
        """
        from modelexpress.megatron_translator import (
            MegatronReceiverContext,
            ReceiveSpec,
            discover_megatron_context,
            run_refit_cycle,
        )
        from modelexpress.nemo_rl_v2 import (
            ROLE_MEGATRON_VOCAB_PARALLEL,
            TargetTpLayout,
        )

        total_start = time.perf_counter()
        mem_start = torch.cuda.memory_allocated(self.device)
        worker_rank = getattr(getattr(self, "_mx_receiver", None), "worker_rank", "?")
        print(
            "[weight-sync-debug][mx-vllm] "
            f"rank={worker_rank} version={version} "
            f"candidates={len(candidates)} megatron_refit_start",
            flush=True,
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
        print(
            "[weight-sync-debug][mx-vllm] "
            f"rank={worker_rank} version={version} "
            f"exact_candidates={len(exact_candidates)} "
            f"stale_candidates={stale_candidates} "
            f"future_candidates={future_candidates}",
            flush=True,
        )
        if not exact_candidates:
            print(
                "[mx-megatron] no exact-version source available for "
                f"version={version}; retrying later"
            )
            return False
        candidates = exact_candidates
        megatron_context_candidates = [
            candidate
            for candidate in candidates
            if getattr(candidate, "megatron_meta", None) is not None
        ]
        if not megatron_context_candidates:
            print(
                "[mx-megatron] no exact-version Megatron trainer metadata "
                f"available for version={version}; retrying later"
            )
            return False

        # ---- One-shot: build context from the first cycle's metadata. ----
        context_s = 0.0
        if not hasattr(self, "_mx_megatron_ctx") or self._mx_megatron_ctx is None:
            context_start = time.perf_counter()
            cfg, name_map = discover_megatron_context(megatron_context_candidates)
            if cfg is None:
                print(
                    "[mx-megatron] sources advertise publisher_kind=megatron but "
                    "no transformer_config sidecar; falling back to non-Megatron "
                    "path on next cycle"
                )
                self._mx_megatron_mode = False
                return False

            # Build receive specs: one per Megatron tensor name in the
            # sidecar's name_map. The receiver's TARGET layout is its own
            # vLLM TP × EP shape.
            target_tp = getattr(self.parallel_config, "tensor_parallel_size", 1)
            target_tp_rank = (
                torch.distributed.get_rank(
                    group=getattr(self, "_tp_process_group", None)
                )
                if torch.distributed.is_initialized()
                else 0
            )
            target_tp_layout = TargetTpLayout(
                tp_size=target_tp,
                tp_rank=target_tp_rank,
            )

            # For each Megatron source-name → list of HF target names, build
            # one ReceiveSpec. Shape + role come from the source's
            # TensorDescriptorV2 in the published shape_registry; the
            # receiver-side parser is in modelexpress.nemo_rl_v2.
            receive_specs: dict[str, ReceiveSpec] = {}
            for cand in megatron_context_candidates:
                if cand.megatron_meta is None or cand.registry is None:
                    continue
                for td in cand.registry.get("tensors", []):
                    if td.name in receive_specs:
                        continue
                    role = td.megatron_role or ""
                    if not role:
                        continue
                    # Bridge's name_map uses unprefixed Megatron names; the
                    # publisher (post-2026-06-08) normalizes to that form, but
                    # be defensive in case a publisher emits the `module.`
                    # prefix and the name_map doesn't.
                    lookup_name = (
                        td.name[len("module.") :]
                        if td.name.startswith("module.")
                        else td.name
                    )
                    hf_names = name_map.get(
                        lookup_name, name_map.get(td.name, [td.name])
                    )
                    receive_specs[td.name] = ReceiveSpec(
                        megatron_name=td.name,
                        hf_names=list(hf_names),
                        role=role,
                        target_shape=tuple(int(s) for s in td.global_shape),
                        target_dtype=td.dtype,
                        shard_axis=int(td.shard_axis),
                        pp_rank=cand.megatron_meta.pp_rank,
                        role_descriptor=dict(td.megatron_extras or {}),
                    )

            self._mx_megatron_ctx = MegatronReceiverContext(
                target_tp_layout=target_tp_layout,
                transformer_config=cfg,
                hf_name_map=name_map,
                receive_specs=receive_specs,
            )
            print(
                f"[mx-megatron] built receive context: tp={target_tp} "
                f"tensors={len(receive_specs)} cfg={cfg}"
            )
            context_s = time.perf_counter() - context_start
            print(
                "[weight-sync-debug][mx-vllm] "
                f"rank={worker_rank} version={version} "
                f"context_build_s={context_s:.3f} "
                f"receive_specs={len(receive_specs)}",
                flush=True,
            )

        # ---- One refit cycle, matched-TP fast path. ----
        # v0: pre-allocate one Megatron-shaped destination per receive_spec,
        # register them all with NIXL under the trainer's Megatron names,
        # and call receive_weights() once to bulk-fill them via a single
        # RDMA pull from the matched-TP-rank source. The translator then
        # walks the filled buffers and produces HF tensors.
        #
        # Mixed-TP (target_tp != source_tp) requires per-source pulls
        # with optional sub-slicing; that's a v1 enhancement landing on
        # top of an MxRefitReceiver.pull_to primitive. Phase B's planner
        # already returns the right slice info; the gap is just the NIXL
        # plumbing for partial-buffer registers.
        ctx = self._mx_megatron_ctx
        if not hasattr(self, "_mx_megatron_buffers"):
            prealloc_start = time.perf_counter()
            prealloc_mem_start = torch.cuda.memory_allocated(self.device)
            buffers: dict[str, "torch.Tensor"] = {}
            vocab_buffers: dict[str, "torch.Tensor"] = {}
            source_tp_size = next(
                (
                    c.megatron_meta.tp_size
                    for c in megatron_context_candidates
                    if c.megatron_meta is not None and c.megatron_meta.tp_size > 0
                ),
                ctx.target_tp_layout.tp_size,
            )
            for spec in ctx.receive_specs.values():
                full_shape = list(spec.target_shape)
                # Replicated tensors stay at full shape; per_expert is
                # one-tensor-per-expert (handled later by run_refit_cycle).
                if spec.role.startswith("expert_"):
                    # Per-expert: skip pre-allocation; per-cycle code path
                    # builds per-expert buffers as part of assembly.
                    continue
                dt = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }.get(spec.target_dtype, torch.bfloat16)
                target = buffers
                if spec.role == ROLE_MEGATRON_VOCAB_PARALLEL:
                    full_shape[int(spec.shard_axis)] *= int(source_tp_size)
                    target = vocab_buffers
                # The Megatron registry shape reflects the published buffer
                # shape. Vocab tensors are the exception: vLLM's loader wants
                # the full vocab tensor and slices it internally for TP.
                target[spec.megatron_name] = torch.empty(
                    full_shape,
                    dtype=dt,
                    device=self.device,
                )
            # Register all at once with the receiver's NIXL plane.
            all_buffers = dict(buffers)
            all_buffers.update(vocab_buffers)
            self._mx_receiver._receiver._nixl.register_tensors(all_buffers)
            self._mx_megatron_buffers = buffers
            self._mx_megatron_vocab_buffers = vocab_buffers
            print(
                f"[mx-megatron] pre-allocated + registered "
                f"{len(buffers)} per-rank Megatron buffers and "
                f"{len(vocab_buffers)} full-vocab buffers "
                f"({sum(b.numel() * b.element_size() for b in all_buffers.values()) / 1e9:.2f} GB)"
            )
            print(
                "[weight-sync-debug][mx-vllm] "
                f"rank={worker_rank} version={version} "
                f"prealloc_register_s={time.perf_counter() - prealloc_start:.3f} "
                f"prealloc_gb="
                f"{sum(b.numel() * b.element_size() for b in all_buffers.values()) / 1e9:.3f} "
                f"cuda_mem_delta_gb="
                f"{(torch.cuda.memory_allocated(self.device) - prealloc_mem_start) / 1e9:.3f}",
                flush=True,
            )
        else:
            print(
                "[weight-sync-debug][mx-vllm] "
                f"rank={worker_rank} version={version} reuse_preallocated_buffers "
                f"buffer_count={len(self._mx_megatron_buffers)} "
                f"vocab_buffer_count={len(getattr(self, '_mx_megatron_vocab_buffers', {}))}",
                flush=True,
            )

        # Choose between matched-TP fast path and mixed-TP per-source path.
        # Matched-TP requires the source's TP-world to equal the receiver's
        # TP-world AND there to exist a source at our tp_rank. Otherwise
        # fall through to the multi-source path.
        matched, selection_stats = _choose_megatron_bulk_source(
            candidates,
            version=int(version),
            target_tp_rank=int(ctx.target_tp_layout.tp_rank),
            target_tp_size=int(ctx.target_tp_layout.tp_size),
        )
        any_megatron_tp_size = next(
            (
                c.megatron_meta.tp_size
                for c in megatron_context_candidates
                if c.megatron_meta is not None and c.megatron_meta.tp_size > 0
            ),
            None,
        )
        target_tp_size = ctx.target_tp_layout.tp_size
        is_matched_tp = matched is not None and (
            any_megatron_tp_size is None or any_megatron_tp_size == target_tp_size
        )
        print(
            "[weight-sync-debug][mx-vllm] "
            f"rank={worker_rank} version={version} "
            f"selection_total={selection_stats['total']} "
            f"selection_same_version={selection_stats['same_version']} "
            f"selection_missing_version={selection_stats['missing_version']} "
            f"selection_stale={selection_stats['stale_version']} "
            f"selection_future={selection_stats['future_version']} "
            f"eligible_trainers={selection_stats['eligible_trainers']} "
            f"eligible_replicas={selection_stats['eligible_replicas']} "
            f"selection_pool={selection_stats['selection_pool']} "
            f"selection_pool_size={selection_stats['selection_pool_size']} "
            f"chosen_role={_mx_candidate_role(matched) if matched is not None else 'none'} "
            f"chosen_worker_rank={getattr(matched, 'worker_rank', 'none') if matched is not None else 'none'} "
            f"chosen_source_id={getattr(getattr(matched, 'ref', None), 'mx_source_id', 'none') if matched is not None else 'none'} "
            f"chosen_worker_id={getattr(getattr(matched, 'ref', None), 'worker_id', 'none') if matched is not None else 'none'} "
            f"chosen_source_key={_mx_candidate_source_key_str(matched)}",
            flush=True,
        )

        weights: list[tuple[str, "torch.Tensor"]] = []

        if is_matched_tp:
            # Bulk RDMA pull — single source, one wire transfer.
            matched_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in self._mx_megatron_buffers.values()
            )
            rebind_start = time.perf_counter()
            self._mx_receiver._receiver._nixl.rebind_tensors(self._mx_megatron_buffers)
            rebind_s = time.perf_counter() - rebind_start
            receive_start = time.perf_counter()
            received_tensors = 0
            for _name, _t in self._mx_receiver.receive_from(
                matched,
                timeout_seconds=mx_config.timeout_seconds,
            ):
                received_tensors += 1
                pass  # buffers filled in-place via NIXL
            receive_s = time.perf_counter() - receive_start
            receive_gbps = (
                (matched_bytes * 8) / (receive_s * 1e9) if receive_s > 0 else 0.0
            )
            print(
                "[weight-sync-debug][mx-vllm] "
                f"rank={worker_rank} version={version} matched_tp_receive "
                f"source_role={_mx_candidate_role(matched)} "
                f"source_rank={matched.worker_rank} "
                f"source_version={_mx_candidate_version(matched)} "
                f"source_id={matched.ref.mx_source_id} "
                f"source_worker_id={matched.ref.worker_id} "
                f"source_key={_mx_candidate_source_key_str(matched)} "
                f"rebind_s={rebind_s:.3f} "
                f"received_tensors={received_tensors} "
                f"receive_gb={matched_bytes / 1e9:.3f} "
                f"receive_s={receive_s:.3f} receive_gbps={receive_gbps:.1f}",
                flush=True,
            )

            vocab_bytes, vocab_slices, vocab_transfer_s, vocab_wall_s = (
                self._mx_pull_megatron_vocab_buffers(
                    candidates=megatron_context_candidates,
                    ctx=ctx,
                    mx_config=mx_config,
                )
            )

            # pre_assembled_buffers tells run_refit_cycle to use the
            # pre-filled tensors instead of calling the per-source pull
            # callback.
            def _noop_pull(src, dest):
                pass

            pre_assembled_buffers = dict(self._mx_megatron_buffers)
            pre_assembled_buffers.update(self._mx_megatron_vocab_buffers)
            translate_start = time.perf_counter()
            for hf_name, hf_tensor in run_refit_cycle(
                self._mx_receiver,
                candidates=megatron_context_candidates,
                context=ctx,
                pull=_noop_pull,
                device=self.device,
                pre_assembled_buffers=pre_assembled_buffers,
            ):
                weights.append((hf_name, hf_tensor))
            translate_s = time.perf_counter() - translate_start
            print(
                "[weight-sync-debug][mx-vllm] "
                f"rank={worker_rank} version={version} matched_tp_translate "
                f"weights={len(weights)} translate_s={translate_s:.3f} "
                f"vocab_gb={vocab_bytes / 1e9:.3f} "
                f"vocab_slices={vocab_slices} "
                f"vocab_transfer_s={vocab_transfer_s:.3f} "
                f"vocab_wall_s={vocab_wall_s:.3f}",
                flush=True,
            )
        else:
            # Mixed-TP (target_tp != source_tp) or per-expert.
            #
            # v1 sliced-pull path:
            #   * Pre-allocate a per-plan dest tensor (target_shape).
            #   * For each plan source whose dest narrow is CONTIGUOUS
            #     (axis 0 — column, qkv, gated_mlp, vocab, per_expert,
            #     passthrough), build a SlicedTransferRequest pointing
            #     directly at the dest view. Each source's requests
            #     batch into one combined NIXL transfer per source.
            #   * For non-contiguous narrows (axis 1 — row-parallel),
            #     fall back to the v0 scratch+host-copy path.
            #
            # Bandwidth profile:
            #   target-narrower: same as v0 (already optimal — each
            #     receiver pulls a different source's full data).
            #   target-wider: v1 cuts wire bytes by target_tp/source_tp×
            #     for axis-0 roles by pulling only the sub-slice each
            #     receiver needs from each source rank. Row-parallel
            #     stays at v0 cost.
            megatron_cands = list(megatron_context_candidates)
            print(
                f"[mx-megatron] mixed-TP path: target_tp={target_tp_size} "
                f"source_tp={any_megatron_tp_size or '?'} "
                f"candidates={len(megatron_cands)}"
            )

            from modelexpress.megatron_translator import (
                translate_megatron_to_hf,
            )
            from modelexpress.nemo_rl_v2 import MegatronTensorSpec

            target_specs: dict[str, MegatronTensorSpec] = {
                m_name: MegatronTensorSpec(
                    role=rs.role,
                    target_shape=rs.target_shape,
                    target_dtype=rs.target_dtype,
                    shard_axis=rs.shard_axis,
                    pp_rank=rs.pp_rank,
                    role_descriptor=dict(rs.role_descriptor or {}),
                )
                for m_name, rs in ctx.receive_specs.items()
            }
            plans = self._mx_receiver.pick_megatron_slice_plans(
                megatron_cands,
                target_tp_layout=ctx.target_tp_layout,
                target_tensor_specs=target_specs,
            )

            # ------ Phase 1: pre-allocate + classify per-plan dests ------
            dt_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            plan_dests: dict[str, "torch.Tensor"] = {}
            # Per-source pull batches: cand_sid -> list[(name, subslice, dest_view)]
            v1_batches: dict[str, list] = {
                c.ref.mx_source_id: [] for c in megatron_cands
            }
            # Plans that need the v0 scratch path (non-contiguous narrows
            # OR per_expert dict assembly).
            v0_plans: list = []

            for plan in plans:
                if not plan.sources:
                    continue
                rs = ctx.receive_specs[plan.tensor_name]
                dt = dt_map.get(rs.target_dtype, torch.bfloat16)
                # per_expert returns a dict; fall back to v0 (host scratch
                # + per-expert assembly).
                if plan.assembly == "per_expert":
                    v0_plans.append(plan)
                    continue
                dest = torch.empty(plan.target_shape, dtype=dt, device=self.device)
                plan_dests[plan.tensor_name] = dest
                axis = 1 if plan.assembly == "concat_dim1" else 0
                routed_to_v1 = True
                for src in plan.sources:
                    target_lo, target_hi = src.target_local_range
                    dest_view = dest.narrow(axis, target_lo, target_hi - target_lo)
                    if not dest_view.is_contiguous():
                        # Row-parallel axis-1 narrow → strided → can't RDMA
                        # directly; whole plan falls back to v0.
                        routed_to_v1 = False
                        break
                    v1_batches[src.mx_source_id].append(
                        (plan.tensor_name, src.source_subslice, dest_view)
                    )
                if not routed_to_v1:
                    # Drop any v1 entries for this plan that we already
                    # queued and route the entire plan to v0.
                    plan_dests.pop(plan.tensor_name, None)
                    for sid in v1_batches:
                        v1_batches[sid] = [
                            r for r in v1_batches[sid] if r[0] != plan.tensor_name
                        ]
                    v0_plans.append(plan)

            n_v1_slices = sum(len(b) for b in v1_batches.values())
            print(
                f"[mx-megatron] v1 sliced-pull: {n_v1_slices} slices across "
                f"{sum(1 for b in v1_batches.values() if b)} sources "
                f"(plans: {len(plan_dests)} via v1, {len(v0_plans)} via v0)"
            )

            # ------ Phase 2: NIXL register the dest buffers ------
            # pull_to writes into the dest_views directly; the underlying
            # buffer must be NIXL-registered so the agent can DMA to it.
            if plan_dests:
                t0 = time.perf_counter()
                self._mx_receiver._receiver._nixl.register_tensors(plan_dests)
                print(
                    f"[mx-megatron] registered {len(plan_dests)} v1 dest buffers "
                    f"with NIXL in {time.perf_counter() - t0:.2f}s"
                )

            # ------ Phase 3: per-source sliced pulls (v1) ------
            t0 = time.perf_counter()
            v1_total_bytes = 0
            for cand in megatron_cands:
                batch = v1_batches[cand.ref.mx_source_id]
                if not batch:
                    continue
                xferred, n_slices, elapsed = self._mx_receiver._receiver.pull_to(
                    cand.ref,
                    batch,
                    timeout_seconds=mx_config.timeout_seconds,
                )
                v1_total_bytes += xferred
            v1_elapsed = time.perf_counter() - t0
            if n_v1_slices:
                v1_bw = (
                    (v1_total_bytes * 8) / (v1_elapsed * 1e9) if v1_elapsed > 0 else 0
                )
                print(
                    f"[mx-megatron] v1 pull complete: {n_v1_slices} slices, "
                    f"{v1_total_bytes / 1e9:.2f} GB, {v1_elapsed:.2f}s, "
                    f"{v1_bw:.1f} Gbps aggregate"
                )

            # ------ Phase 4: v0 fallback for plans that needed scratch ------
            scratch: dict[str, dict[str, "torch.Tensor"]] = {}
            if v0_plans:
                # Identify which sources contribute to the v0 plans.
                v0_source_ids = set()
                for plan in v0_plans:
                    for src in plan.sources:
                        v0_source_ids.add(src.mx_source_id)
                v0_cands = [
                    c for c in megatron_cands if c.ref.mx_source_id in v0_source_ids
                ]
                t0 = time.perf_counter()
                for cand in v0_cands:
                    buf_dict: dict[str, "torch.Tensor"] = {}
                    for name, t in self._mx_receiver._receiver.receive_weights_scratch(
                        cand.ref,
                        timeout_seconds=mx_config.timeout_seconds,
                    ):
                        buf_dict[name] = t
                    scratch[cand.ref.mx_source_id] = buf_dict
                print(
                    f"[mx-megatron] v0 fallback: scratch-pulled {len(v0_plans)} "
                    f"plans from {len(v0_cands)} sources in "
                    f"{time.perf_counter() - t0:.2f}s"
                )
            # ------ Phase 5: assemble + translate per plan ------
            from modelexpress.megatron_translator import (
                assemble_into_destination,
            )

            for plan in plans:
                if not plan.sources:
                    continue
                rs = ctx.receive_specs[plan.tensor_name]
                if plan.tensor_name in plan_dests:
                    # v1 path: dest was filled directly by the sliced
                    # NIXL transfer. No host-side assembly needed.
                    assembled = plan_dests[plan.tensor_name]
                else:
                    # v0 path: scratch+slice-copy via assemble_into_destination
                    # with a name+source-aware pull callback.
                    def _pull_factory(name=plan.tensor_name, assembly=plan.assembly):
                        def _pull(src, dest):
                            full = scratch.get(src.mx_source_id, {}).get(name)
                            if full is None:
                                raise RuntimeError(
                                    f"mixed-TP v0: scratch missing {name!r} from "
                                    f"source {src.mx_source_id}"
                                )
                            axis = 1 if assembly == "concat_dim1" else 0
                            if src.source_subslice is not None:
                                slo, shi = src.source_subslice
                                slice_src = full.narrow(axis, slo, shi - slo)
                            else:
                                slice_src = full
                            if slice_src.shape != dest.shape:
                                raise RuntimeError(
                                    f"mixed-TP v0 shape mismatch on {name}: "
                                    f"src={tuple(slice_src.shape)} "
                                    f"dest={tuple(dest.shape)} "
                                    f"axis={axis} subslice={src.source_subslice}"
                                )
                            dest.copy_(slice_src, non_blocking=True)

                        return _pull

                    assembled = assemble_into_destination(
                        plan,
                        pull=_pull_factory(),
                        device=self.device,
                    )
                for hf_name, hf_tensor in translate_megatron_to_hf(
                    plan,
                    assembled,
                    transformer_config=ctx.transformer_config,
                    hf_names=list(rs.hf_names),
                ):
                    weights.append((hf_name, hf_tensor))

        if not weights:
            print("[mx-megatron] cycle yielded 0 tensors; refit aborted")
            return False

        load_start = time.perf_counter()
        self._load_weights(weights)
        load_s = time.perf_counter() - load_start
        sync_start = time.perf_counter()
        torch.cuda.current_stream().synchronize()
        sync_s = time.perf_counter() - sync_start
        fp8_start = time.perf_counter()
        self._maybe_process_fp8_kv_cache()
        fp8_s = time.perf_counter() - fp8_start

        publish_self_s = 0.0
        if mx_config.tree_scale_out:
            publish_self_start = time.perf_counter()
            self._mx_receiver.publish_self_as_source(
                version=int(version),
                model_name=self.model_config.model
                if hasattr(self.model_config, "model")
                else getattr(
                    self.model_runner.vllm_config.model_config,
                    "model",
                    "unknown",
                ),
            )
            publish_self_s = time.perf_counter() - publish_self_start

        gc_start = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        gc_s = time.perf_counter() - gc_start
        mem_end = torch.cuda.memory_allocated(self.device)
        print(
            "[weight-sync-debug][mx-vllm] "
            f"rank={worker_rank} version={version} "
            f"load_s={load_s:.3f} cuda_sync_s={sync_s:.3f} "
            f"fp8_s={fp8_s:.3f} publish_self_s={publish_self_s:.3f} "
            f"gc_empty_cache_s={gc_s:.3f} "
            f"cuda_mem_delta_gb={(mem_end - mem_start) / 1e9:.3f} "
            f"total_s={time.perf_counter() - total_start:.3f}",
            flush=True,
        )
        self._mx_loaded_version = int(version)
        return True

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        load_model_weight_func = self._load_weights
        total_start = time.perf_counter()
        mem_start = torch.cuda.memory_allocated(self.device)

        try:
            collective_start = time.perf_counter()
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )
            collective_s = time.perf_counter() - collective_start

            # Process weights after loading for FP8 KV cache
            fp8_start = time.perf_counter()
            self._maybe_process_fp8_kv_cache()
            fp8_s = time.perf_counter() - fp8_start
            print(
                "[weight-sync-debug][nccl-vllm] "
                f"collective_and_load_s={collective_s:.3f} fp8_s={fp8_s:.3f} "
                f"cuda_mem_delta_gb="
                f"{(torch.cuda.memory_allocated(self.device) - mem_start) / 1e9:.3f} "
                f"total_s={time.perf_counter() - total_start:.3f}",
                flush=True,
            )

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
