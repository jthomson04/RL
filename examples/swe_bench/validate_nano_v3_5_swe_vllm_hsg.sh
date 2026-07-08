#!/usr/bin/env bash
# Validate the regular NeMo-RL vLLM side of the HSG SWE comparison image.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}
PERSISTENT_CACHE=${PERSISTENT_CACHE:-${ROOT}/cache/nemotron_nano_v3_5_vllm020}
NEMO_RL_VENV_DIR=${VLLM_COMPARISON_VENV_DIR:-${PERSISTENT_CACHE}/venvs}
VLLM_ACTOR=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
VLLM_VENV=${NEMO_RL_VENV_DIR}/${VLLM_ACTOR}
UV_CACHE_DIR=${UV_CACHE_DIR:-${PERSISTENT_CACHE}/uv}

cd "${REPO_ROOT}"
mkdir -p "${NEMO_RL_VENV_DIR}" "${UV_CACHE_DIR}"

if [[ ! -e "${VLLM_VENV}/bin/python" ]] || \
  ! "${VLLM_VENV}/bin/python" -c 'import vllm' >/dev/null 2>&1; then
  UV_PROJECT_ENVIRONMENT="${VLLM_VENV}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    uv sync --locked --extra vllm --directory "${REPO_ROOT}"
fi

"${VLLM_VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import inspect
from pathlib import Path

import yaml
import vllm.tool_parsers  # noqa: F401 - registers built-in parsers
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.utils.import_utils import import_from_path

assert metadata.version("vllm") == "0.20.0"

reasoning_plugin = Path(
    "nemo_rl/models/generation/vllm/reasoning_parsers/"
    "nano_v3_reasoning_parser.py"
).resolve()
import_from_path("nano_v3_reasoning_parser_preflight", reasoning_plugin)
assert "nano_v3" in ReasoningParserManager.reasoning_parsers
ReasoningParserManager.reasoning_parsers.pop("nano_v3")
ReasoningParserManager.import_reasoning_parser(str(reasoning_plugin))

ToolParserManager.get_tool_parser("qwen3_coder")
assert "qwen3_coder" in ToolParserManager.tool_parsers
assert "nano_v3" in ReasoningParserManager.reasoning_parsers

with Path(
    "examples/swe_bench/grpo_nano_v3_5_swe_vllm_hsg.yaml"
).open(encoding="utf-8") as config_file:
    generation = yaml.safe_load(config_file)["policy"]["generation"]

assert generation["backend"] == "vllm"
assert "dynamo_cfg" not in generation
assert generation["vllm_cfg"]["async_engine"] is True
assert generation["vllm_cfg"]["expose_http_server"] is True
engine_args = inspect.signature(AsyncEngineArgs).parameters
unsupported_kwargs = sorted(set(generation["vllm_kwargs"]) - set(engine_args))
assert not unsupported_kwargs, f"unsupported vLLM 0.20 arguments: {unsupported_kwargs}"

print("regular vLLM", metadata.version("vllm"))
print("tool parser qwen3_coder: registered")
print("reasoning parser nano_v3: registered")
print("regular-vLLM SWE configuration: validated")
PY

"${VLLM_VENV}/bin/python" -m pytest \
  tests/unit/distributed/test_ray_actor_environment_registry.py \
  tests/unit/models/generation/test_swe_backend_comparison_config.py \
  tests/unit/models/generation/test_vllm_generation.py \
  -q

/opt/nemo_rl_venv/bin/python -m ruff check \
  nemo_rl/distributed/ray_actor_environment_registry.py \
  tests/unit/distributed/test_ray_actor_environment_registry.py \
  tests/unit/models/generation/test_swe_backend_comparison_config.py
/opt/nemo_rl_venv/bin/python -m ruff format --check \
  nemo_rl/distributed/ray_actor_environment_registry.py \
  tests/unit/distributed/test_ray_actor_environment_registry.py \
  tests/unit/models/generation/test_swe_backend_comparison_config.py
