#!/usr/bin/env bash
# Validate the regular NeMo-RL vLLM side of the HSG SWE comparison image.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}
PERSISTENT_CACHE=${PERSISTENT_CACHE:-${ROOT}/cache/nemotron_nano_v3_5_vllm023}
NEMO_RL_VENV_DIR=${VLLM_COMPARISON_VENV_DIR:-${PERSISTENT_CACHE}/venvs}
VLLM_ACTOR=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
VLLM_VENV=${NEMO_RL_VENV_DIR}/${VLLM_ACTOR}
UV_CACHE_DIR=${UV_CACHE_DIR:-${PERSISTENT_CACHE}/uv}
STACK_MARKER=${VLLM_VENV}/.dynamo-vllm-stack
EXPECTED_STACK_MARKER='vllm=0.23.0 python=3.13 backport=vllm#44814@45ffb397'

cd "${REPO_ROOT}"
mkdir -p "${NEMO_RL_VENV_DIR}" "${UV_CACHE_DIR}"

if [[ ! -e "${VLLM_VENV}/bin/python" ]] || \
  ! "${VLLM_VENV}/bin/python" -c 'import vllm' >/dev/null 2>&1; then
  UV_PROJECT_ENVIRONMENT="${VLLM_VENV}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    uv sync --locked --extra vllm --directory "${REPO_ROOT}"
fi

# Ray requires the actor and driver to use the same Python minor version. The
# image's NeMo-RL/Ray environment is Python 3.13, whereas /opt/dynamo_venv is
# Python 3.12. Resolve vLLM 0.23's dependencies into the actor's Python 3.13
# environment, then apply the same NemotronH layerwise-reload backport used in
# /opt/dynamo_venv. We cannot link Dynamo's entire Python 3.12 distribution:
# although vLLM's primary extensions use the stable ABI, some optional modules
# are CPython-minor-specific. The patched refit implementation is compared
# byte-for-byte below so the behavior relevant to this workload cannot drift.
if [[ ! -f "${STACK_MARKER}" ]] || \
  [[ "$(<"${STACK_MARKER}")" != "${EXPECTED_STACK_MARKER}" ]]; then
  ACTOR_SITE=$(
    "${VLLM_VENV}/bin/python" -c \
      'import sysconfig; print(sysconfig.get_paths()["purelib"])'
  )
  rm -f "${STACK_MARKER}"
  [[ ! -L "${ACTOR_SITE}/vllm" ]] || rm "${ACTOR_SITE}/vllm"
  [[ ! -L "${ACTOR_SITE}/vllm-0.23.0.dist-info" ]] || \
    rm "${ACTOR_SITE}/vllm-0.23.0.dist-info"
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    uv pip install --python "${VLLM_VENV}/bin/python" --upgrade 'vllm==0.23.0'

  VLLM_RELOAD_PATCH=${REPO_ROOT}/docker/patches/vllm-0.23.0-layerwise-reload-composed-loader.patch
  test -f "${VLLM_RELOAD_PATCH}"
  (
    cd "${ACTOR_SITE}"
    if ! git apply --reverse --check "${VLLM_RELOAD_PATCH}"; then
      git apply --check "${VLLM_RELOAD_PATCH}"
      git apply "${VLLM_RELOAD_PATCH}"
    fi
  )
  printf '%s\n' "${EXPECTED_STACK_MARKER}" > "${STACK_MARKER}"
fi

"${VLLM_VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import inspect
import json
import subprocess
from pathlib import Path

import yaml
import vllm
import vllm.tool_parsers  # noqa: F401 - registers built-in parsers
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.utils.import_utils import import_from_path

assert metadata.version("vllm") == "0.23.0"
assert not Path(vllm.__file__).resolve().is_relative_to(
    Path("/opt/dynamo_venv").resolve()
), vllm.__file__
assert Path("/opt/vllm_backports").read_text(encoding="utf-8").strip() == (
    "vllm#44814 45ffb397d1c7803a78c32846807c71d881e11189"
)
actor_reload_meta = (
    Path(vllm.__file__).resolve().parent
    / "model_executor/model_loader/reload/meta.py"
)
dynamo_reload_meta = (
    Path("/opt/dynamo_venv/lib/python3.12/site-packages/vllm")
    / "model_executor/model_loader/reload/meta.py"
)
assert actor_reload_meta.read_bytes() == dynamo_reload_meta.read_bytes(), (
    actor_reload_meta,
    dynamo_reload_meta,
)

comparison_packages = (
    "vllm",
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
    "transformers",
    "tokenizers",
    "flashinfer-python",
    "flashinfer-cubin",
    "compressed-tensors",
)
version_probe = (
    "import importlib.metadata as m, json, sys; "
    "print(json.dumps({name: m.version(name) for name in sys.argv[1:]}))"
)
dynamo_versions = json.loads(
    subprocess.check_output(
        ["/opt/dynamo_venv/bin/python", "-c", version_probe, *comparison_packages],
        text=True,
    )
)
actor_versions = {name: metadata.version(name) for name in comparison_packages}
assert actor_versions == dynamo_versions, (
    "regular-vLLM actor stack differs from Dynamo: ",
    actor_versions,
    dynamo_versions,
)

reasoning_plugin = Path(
    "nemo_rl/models/generation/vllm/reasoning_parsers/"
    "nano_v3_reasoning_parser.py"
).resolve()
import_from_path("nano_v3_reasoning_parser_preflight", reasoning_plugin)
ReasoningParserManager.get_reasoning_parser("nano_v3")
assert "nano_v3" in ReasoningParserManager.reasoning_parsers
ReasoningParserManager.reasoning_parsers.pop("nano_v3")
ReasoningParserManager.lazy_parsers.pop("nano_v3", None)
ReasoningParserManager.import_reasoning_parser(str(reasoning_plugin))
ReasoningParserManager.get_reasoning_parser("nano_v3")

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
assert not unsupported_kwargs, f"unsupported vLLM 0.23 arguments: {unsupported_kwargs}"

print("regular vLLM", metadata.version("vllm"))
print("vLLM source", Path(vllm.__file__).resolve())
print("matched inference packages", json.dumps(actor_versions, sort_keys=True))
print("vLLM NemotronH refit backport: validated")
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
