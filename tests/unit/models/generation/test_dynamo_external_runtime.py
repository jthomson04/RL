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

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_rl.models.generation.dynamo import (
    DynamoConfig,
    DynamoGeneration,
    DynamoGraphDeploymentCfg,
)
from nemo_rl.models.generation.dynamo import external_runtime
from nemo_rl.utils import k8s as k8s_utils
from nemo_rl.utils.config import load_config


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DYNAMO_K8S_RECIPES = sorted(
    path
    for path in (_REPO_ROOT / "infra/nrl_k8s/examples/dynamo").glob("V*/grpo*.yaml")
    if ".infra." not in path.name
)


def _external_config(**dynamo_overrides: Any) -> dict[str, Any]:
    dynamo_cfg = {
        "engine_world_size": 2,
        "dgd_name": "test-dgd",
        "frontend_url": None,
        "namespace": "training",
        "frontend_port": 8000,
        "dyn_system_port": 9090,
        "request_timeout_s": 30,
        "discovery_timeout_s": 5,
        "control_timeout_s": 10,
        "exclude_tools_when_tool_choice_none": True,
        "metrics_include_prefixes": None,
        "metrics_exclude_prefixes": None,
    }
    dynamo_cfg.update(dynamo_overrides)
    return {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {"enabled": False},
        "dynamo_cfg": dynamo_cfg,
        "vllm_cfg": {
            "async_engine": True,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": 512,
            "load_format": "auto",
            "enforce_eager": False,
            "expose_http_server": False,
            "enable_vllm_metrics_logger": False,
            "vllm_metrics_logger_interval": 1.0,
            "env_vars": None,
        },
        "vllm_kwargs": {},
    }


def test_config_selects_strict_external_shape() -> None:
    validated = DynamoConfig.model_validate(_external_config())

    assert isinstance(validated.dynamo_cfg, DynamoGraphDeploymentCfg)
    assert validated.engine_world_size == 2

    config = _external_config(unknown_field=True)
    with pytest.raises(ValidationError, match="unknown_field"):
        DynamoConfig.model_validate(config)

    config = _external_config(engine_world_size=3)
    with pytest.raises(ValidationError, match="engine_world_size must equal"):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize("recipe_path", _DYNAMO_K8S_RECIPES)
def test_k8s_recipe_resolves_against_dynamo_schema(recipe_path: Path) -> None:
    assert len(_DYNAMO_K8S_RECIPES) == 4
    config = load_config(recipe_path)
    config.policy.generation.dynamo_cfg.dgd_name = "schema-check"

    validated = DynamoConfig.model_validate(
        OmegaConf.to_container(config.policy.generation, resolve=True)
    )

    assert isinstance(validated.dynamo_cfg, DynamoGraphDeploymentCfg)


def test_frontend_resolution_uses_operator_service_name(monkeypatch) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    cfg = DynamoGraphDeploymentCfg.model_validate(_external_config()["dynamo_cfg"])

    assert external_runtime._resolve_frontend_url(cfg) == (
        "http://test-dgd-frontend.training.svc.cluster.local:8000/v1"
    )


def test_read_pod_namespace(tmp_path: Path, monkeypatch) -> None:
    namespace_file = tmp_path / "namespace"
    namespace_file.write_text("training\n")
    monkeypatch.setattr(k8s_utils, "_POD_NAMESPACE_FILE", namespace_file)

    assert k8s_utils.read_pod_namespace() == "training"

    namespace_file.unlink()
    assert k8s_utils.read_pod_namespace() is None


def test_explicit_frontend_url_works_outside_kubernetes(monkeypatch) -> None:
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = DynamoGraphDeploymentCfg.model_validate(
        _external_config(
            dgd_name=None,
            frontend_url="https://dynamo.example.test/v1",
            namespace=None,
        )["dynamo_cfg"]
    )

    assert (
        external_runtime._resolve_frontend_url(cfg) == "https://dynamo.example.test/v1"
    )


def test_frontend_resolution_requires_kubernetes(monkeypatch) -> None:
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    cfg = DynamoGraphDeploymentCfg.model_validate(_external_config()["dynamo_cfg"])

    with pytest.raises(RuntimeError, match="requires running inside"):
        external_runtime._resolve_frontend_url(cfg)


def test_worker_discovery_filters_deduplicates_and_sorts(monkeypatch) -> None:
    payload = {
        "instances": [
            {
                "namespace": "training-test-dgd",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "worker-b",
                "transport": {"tcp": "tcp://10.0.0.3:5555/channel/rl"},
            },
            {
                "namespace": "training-test-dgd",
                "component": "backend",
                "endpoint": "generate",
                "instance_id": "worker-b",
                "transport": {"tcp": "tcp://10.0.0.3:5555/channel/generate"},
            },
            {
                "namespace": "other",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "wrong-namespace",
                "transport": {"tcp": "tcp://10.0.0.9:5555/channel/rl"},
            },
            {
                "namespace": "training-test-dgd",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "worker-a",
                "transport": {"tcp": "10.0.0.2:5555/channel/rl"},
            },
        ]
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        external_runtime.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(),
    )

    assert external_runtime._discover_worker_instances(
        frontend_host="test-dgd-frontend.training.svc.cluster.local",
        frontend_port=8000,
        dyn_namespaces={"training-test-dgd"},
        dyn_system_port=9090,
        timeout_s=5,
    ) == [
        {"instance_id": "worker-a", "system_url": "http://10.0.0.2:9090"},
        {"instance_id": "worker-b", "system_url": "http://10.0.0.3:9090"},
    ]


def test_external_runtime_rejects_membership_change(monkeypatch) -> None:
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    first = [{"instance_id": "a", "system_url": "http://10.0.0.2:9090"}]
    second = [{"instance_id": "b", "system_url": "http://10.0.0.3:9090"}]
    discoveries = iter([first, second])
    monkeypatch.setattr(
        external_runtime,
        "_discover_worker_instances",
        lambda **kwargs: next(discoveries),
    )

    runtime = external_runtime.ExternalDynamoRuntime(config=_external_config())
    expected = runtime.refit_workers()
    with pytest.raises(RuntimeError, match="membership changed"):
        runtime.validate_workers(expected)


def test_generation_accepts_no_ray_cluster_for_external_runtime(
    monkeypatch,
) -> None:
    events: list[str] = []

    class FakeExternalRuntime:
        def __init__(self, *, config):
            events.append("init")

        @property
        def frontend_url(self):
            return "http://test-dgd-frontend.training.svc.cluster.local:8000/v1"

        def start(self):
            events.append("start")

        def refit_workers(self):
            return [{"instance_id": "worker", "system_url": "http://10.0.0.2:9090"}]

        def validate_workers(self, expected):
            return expected

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(
        external_runtime,
        "ExternalDynamoRuntime",
        FakeExternalRuntime,
    )

    generation = DynamoGeneration(cluster=None, config=_external_config())

    assert generation.get_inference_world_size() == 2
    assert (
        generation.frontend_url
        == "http://test-dgd-frontend.training.svc.cluster.local:8000/v1"
    )
    assert generation.dp_openai_server_base_urls == [None]
    assert generation.shutdown()
    assert generation.shutdown()
    assert events == ["init", "start", "shutdown"]
