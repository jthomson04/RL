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

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    VLLM_EXECUTABLE,
    _resolve_vllm_executable,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


def test_vllm_executable_defaults_follow_global_system_selection(monkeypatch) -> None:
    monkeypatch.delenv("NEMO_RL_VLLM_PY_EXECUTABLE", raising=False)

    assert _resolve_vllm_executable(True) == PY_EXECUTABLES.SYSTEM
    assert _resolve_vllm_executable(False) == PY_EXECUTABLES.VLLM


def test_vllm_executable_override_takes_precedence(monkeypatch) -> None:
    override = "/lustre/cache/vllm023/venvs/vllm-worker/bin/python"
    monkeypatch.setenv("NEMO_RL_VLLM_PY_EXECUTABLE", override)

    assert _resolve_vllm_executable(True) == override
    assert _resolve_vllm_executable(False) == override


def test_complete_vllm_actor_family_uses_one_environment() -> None:
    actor_fqns = (
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
        "nemo_rl.algorithms.async_utils.ReplayBuffer",
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
    )

    assert {
        ACTOR_ENVIRONMENT_REGISTRY[actor_fqn] for actor_fqn in actor_fqns
    } == {VLLM_EXECUTABLE}
