#!/usr/bin/env bash
# Read-only preflight for the regular NeMo-RL vLLM side of the HSG SWE run.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}
PERSISTENT_CACHE=${PERSISTENT_CACHE:-${ROOT}/cache/nemotron_nano_v3_5_vllm023}
NEMO_RL_VENV_DIR=${VLLM_COMPARISON_VENV_DIR:-${PERSISTENT_CACHE}/venvs}
VLLM_ACTOR=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
VLLM_VENV=${NEMO_RL_VENV_DIR}/${VLLM_ACTOR}
MANIFEST_DIR=${VLLM_ACTOR_MANIFEST_DIR:-${PERSISTENT_CACHE}/manifests}
FREEZE_FILE=${MANIFEST_DIR}/vllm023-actor-freeze.txt
FREEZE_SHA_FILE=${FREEZE_FILE}.sha256
STACK_MARKER=${VLLM_VENV}/.dynamo-vllm-stack

cd "${REPO_ROOT}"
test -x "${VLLM_VENV}/bin/python"
test -s "${STACK_MARKER}"
test -s "${FREEZE_FILE}"
test -s "${FREEZE_SHA_FILE}"
(
  cd "${MANIFEST_DIR}"
  sha256sum --check "$(basename "${FREEZE_SHA_FILE}")"
)

REPO_ROOT="${REPO_ROOT}" \
  "${VLLM_VENV}/bin/python" \
  examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py

NEMO_RL_VLLM_PY_EXECUTABLE="${VLLM_VENV}/bin/python" \
  /opt/nemo_rl_venv/bin/python -m pytest \
  tests/unit/distributed/test_ray_actor_environment_registry.py \
  tests/unit/models/generation/test_swe_backend_comparison_config.py \
  tests/unit/models/generation/test_vllm_generation.py::test_resolve_enable_prefix_caching_respects_explicit_config \
  tests/unit/models/generation/test_vllm_generation.py::test_resolve_enable_prefix_caching_uses_cuda_capability_for_auto \
  tests/unit/models/generation/test_vllm_generation.py::test_vllm_async_http_server_loads_reasoning_parser_plugin \
  tests/unit/models/generation/test_vllm_generation.py::test_nano_v3_reasoning_parser_swaps_reasoning_when_thinking_disabled \
  tests/unit/models/generation/test_vllm_generation.py::test_configure_generation_config_uses_real_startup_weights_without_draft_refit \
  tests/unit/models/generation/test_vllm_generation.py::test_configure_generation_config_keeps_dummy_startup_weights_with_draft_refit \
  tests/unit/models/generation/test_vllm_generation.py::test_configure_generation_config_keeps_dummy_startup_weights_for_mtp \
  -q

/opt/nemo_rl_venv/bin/python -m ruff check \
  nemo_rl/distributed/ray_actor_environment_registry.py \
  tests/unit/distributed/test_ray_actor_environment_registry.py \
  tests/unit/models/generation/test_swe_backend_comparison_config.py \
  examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py
/opt/nemo_rl_venv/bin/python -m ruff format --check \
  nemo_rl/distributed/ray_actor_environment_registry.py \
  tests/unit/distributed/test_ray_actor_environment_registry.py \
  tests/unit/models/generation/test_swe_backend_comparison_config.py \
  examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py
