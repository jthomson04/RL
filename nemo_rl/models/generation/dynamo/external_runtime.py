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

"""Client runtime for an externally owned DynamoGraphDeployment."""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from nemo_rl.models.generation.dynamo.config import (
    DynamoConfig,
    DynamoGraphDeploymentCfg,
)
from nemo_rl.utils.k8s import is_in_kubernetes, read_pod_namespace

LOGGER = logging.getLogger(__name__)

_HTTP_MAX_ATTEMPTS = 3
_HTTP_RETRY_DELAY_S = 1.0
_RETRYABLE_HTTP_STATUS_CODES = {408, 429}


class _WorkerDiscoveryError(RuntimeError):
    """Dynamo worker discovery failed without confirming a fleet change."""


class _RetryableWorkerDiscoveryError(_WorkerDiscoveryError):
    """Dynamo worker discovery failed in a way that may be transient."""


def _discover_worker_instances(
    *,
    frontend_host: str,
    frontend_port: int,
    dyn_namespaces: set[str],
    dyn_system_port: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Discover fixed vLLM worker endpoints through the frontend health route."""
    url = f"http://{frontend_host}:{frontend_port}/health"
    data: Any = None
    for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
        error: _WorkerDiscoveryError | None = None
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            error_type = (
                _RetryableWorkerDiscoveryError
                if exc.code in _RETRYABLE_HTTP_STATUS_CODES or 500 <= exc.code < 600
                else _WorkerDiscoveryError
            )
            error = error_type(
                f"Dynamo worker discovery request to {url} failed with HTTP {exc.code}."
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            error = _RetryableWorkerDiscoveryError(
                f"Dynamo worker discovery request to {url} failed: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            try:
                data = json.loads(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                error = _RetryableWorkerDiscoveryError(
                    f"Dynamo worker discovery response from {url} was not valid JSON."
                )
            else:
                break

        assert error is not None
        if not isinstance(error, _RetryableWorkerDiscoveryError):
            raise error
        if attempt == _HTTP_MAX_ATTEMPTS:
            raise _WorkerDiscoveryError(
                f"Dynamo worker discovery through {url} failed after "
                f"{_HTTP_MAX_ATTEMPTS} attempts: {error}"
            ) from error
        LOGGER.warning(
            "Dynamo worker discovery attempt %d/%d failed; retrying in %.1fs: %s",
            attempt,
            _HTTP_MAX_ATTEMPTS,
            _HTTP_RETRY_DELAY_S,
            error,
        )
        time.sleep(_HTTP_RETRY_DELAY_S)

    if not isinstance(data, dict):
        raise _WorkerDiscoveryError(
            f"Dynamo worker discovery response from {url} was not a JSON object."
        )
    instances = data.get("instances", [])
    if not isinstance(instances, list):
        raise _WorkerDiscoveryError(
            f"Dynamo worker discovery response from {url} had a non-list "
            "instances field."
        )

    seen_by_id: dict[Any, dict[str, Any]] = {}
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        if instance.get("namespace") not in dyn_namespaces:
            continue
        if instance.get("component") != "backend":
            continue
        if instance.get("endpoint") != "rl":
            continue

        instance_id = instance.get("instance_id")
        if instance_id is None:
            raise _WorkerDiscoveryError(
                "Dynamo worker discovery returned an rl backend without an instance_id."
            )
        transport = instance.get("transport") or {}
        tcp = transport.get("tcp") if isinstance(transport, dict) else None
        if not isinstance(tcp, str):
            raise _WorkerDiscoveryError(
                f"Dynamo worker {instance_id!r} did not advertise a TCP transport URL."
            )

        transport_url = tcp if "://" in tcp else f"tcp://{tcp}"
        try:
            parsed_transport = urllib.parse.urlsplit(transport_url)
            pod_ip = parsed_transport.hostname
            transport_port = parsed_transport.port
        except ValueError as exc:
            raise _WorkerDiscoveryError(
                f"Dynamo worker {instance_id!r} advertised invalid TCP transport "
                f"URL {tcp!r}."
            ) from exc
        if pod_ip is None or transport_port is None:
            raise _WorkerDiscoveryError(
                f"Dynamo worker {instance_id!r} advertised invalid TCP transport "
                f"URL {tcp!r}."
            )
        system_host = f"[{pod_ip}]" if ":" in pod_ip else pod_ip
        seen_by_id.setdefault(
            instance_id,
            {
                "instance_id": instance_id,
                "system_url": f"http://{system_host}:{dyn_system_port}",
            },
        )
    return sorted(seen_by_id.values(), key=lambda worker: str(worker["instance_id"]))


def _resolve_dgd_namespace(dynamo_cfg: DynamoGraphDeploymentCfg) -> str:
    """Resolve the DGD namespace without guessing a potentially wrong value."""
    namespace = dynamo_cfg.namespace or read_pod_namespace()
    if namespace:
        return namespace
    raise RuntimeError(
        "Could not determine the DynamoGraphDeployment namespace. Set "
        "policy.generation.dynamo_cfg.namespace explicitly or enable the "
        "Kubernetes service-account namespace projection."
    )


def _derive_frontend_url_from_dgd(
    dynamo_cfg: DynamoGraphDeploymentCfg,
) -> str:
    """Build the cluster-internal URL of the DGD frontend Service."""
    dgd_name = dynamo_cfg.dgd_name
    assert dgd_name is not None
    namespace = _resolve_dgd_namespace(dynamo_cfg)
    return (
        f"http://{dgd_name}-frontend.{namespace}.svc.cluster.local:"
        f"{dynamo_cfg.frontend_port}/v1"
    )


def _resolve_frontend_url(dynamo_cfg: DynamoGraphDeploymentCfg) -> str:
    """Resolve an explicit or operator-derived frontend URL."""
    if dynamo_cfg.frontend_url is not None:
        if not dynamo_cfg.frontend_url:
            raise RuntimeError(
                "policy.generation.dynamo_cfg.frontend_url is set but empty."
            )
        return dynamo_cfg.frontend_url

    if dynamo_cfg.dgd_name is None:
        raise RuntimeError(
            "External Dynamo requires either policy.generation.dynamo_cfg.dgd_name "
            "or policy.generation.dynamo_cfg.frontend_url."
        )
    if not is_in_kubernetes():
        raise RuntimeError(
            "External Dynamo with dgd_name requires running inside a Kubernetes "
            "pod. Set policy.generation.dynamo_cfg.frontend_url when running "
            "outside Kubernetes."
        )
    return _derive_frontend_url_from_dgd(dynamo_cfg)


class ExternalDynamoRuntime:
    """Resolve and monitor a DGD without taking ownership of its lifecycle."""

    def __init__(self, *, config: dict[str, Any]) -> None:
        validated_config = DynamoConfig.model_validate(config)
        dynamo_cfg = validated_config.dynamo_cfg
        if not isinstance(dynamo_cfg, DynamoGraphDeploymentCfg):
            raise TypeError("ExternalDynamoRuntime requires DynamoGraphDeploymentCfg.")
        self._dynamo_cfg = dynamo_cfg
        self._frontend_url = _resolve_frontend_url(dynamo_cfg)
        self._workers: list[dict[str, Any]] | None = None
        self._discovery_kwargs: dict[str, Any] | None = None

    @property
    def frontend_url(self) -> str:
        """Return the externally managed frontend URL."""
        return self._frontend_url

    def start(self) -> None:
        """Leave lifecycle management to nrl-k8s."""

    def _build_worker_discovery_kwargs(self) -> dict[str, Any]:
        dgd_name = self._dynamo_cfg.dgd_name
        if dgd_name is None:
            raise RuntimeError(
                "Dynamo NCCL weight transfer requires "
                "policy.generation.dynamo_cfg.dgd_name."
            )
        namespace = _resolve_dgd_namespace(self._dynamo_cfg)
        return {
            "frontend_host": f"{dgd_name}-frontend.{namespace}.svc.cluster.local",
            "frontend_port": self._dynamo_cfg.frontend_port,
            "dyn_namespaces": {f"{namespace}-{dgd_name}"},
            "dyn_system_port": self._dynamo_cfg.dyn_system_port,
            "timeout_s": self._dynamo_cfg.discovery_timeout_s,
        }

    def refit_workers(self) -> list[dict[str, Any]]:
        """Discover and freeze the fixed DGD worker fleet."""
        if self._workers is not None:
            return [dict(worker) for worker in self._workers]
        self._discovery_kwargs = self._build_worker_discovery_kwargs()
        workers = _discover_worker_instances(**self._discovery_kwargs)
        if not workers:
            raise RuntimeError(
                "No Dynamo vLLM workers advertising the rl endpoint were "
                "discovered through the DGD frontend /health response."
            )
        self._workers = workers
        return [dict(worker) for worker in workers]

    def validate_workers(self, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fail if the externally owned worker fleet changed after setup."""
        if self._discovery_kwargs is None:
            self._discovery_kwargs = self._build_worker_discovery_kwargs()
        current = _discover_worker_instances(**self._discovery_kwargs)
        expected_ids = [
            (worker["instance_id"], worker["system_url"]) for worker in expected
        ]
        current_ids = [
            (worker["instance_id"], worker["system_url"]) for worker in current
        ]
        if current_ids != expected_ids:
            raise RuntimeError(
                "Dynamo worker membership changed after NCCL collective "
                f"initialization: expected={expected_ids}, current={current_ids}. "
                "Restart the training job to establish a new fixed collective."
            )
        return current

    def shutdown(self) -> None:
        """Leave the external DGD running for nrl-k8s to clean up."""
