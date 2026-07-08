#!/usr/bin/env bash
# Read-only parser, import, manifest, and Ray startup validation for the baked image.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/opt/nemo-rl}
VLLM_VENV=/opt/nemo_vllm_venv
TARGET_FREEZE=/opt/nemo_vllm_actor_freeze.txt
TARGET_RECORD_HASHES=/opt/nemo_vllm_actor_record_hashes.txt

test "$(uname -m)" = aarch64
test -x "${VLLM_VENV}/bin/python"
test -s "${VLLM_VENV}/.dynamo-vllm-stack"
test -s "${TARGET_FREEZE}"
test -s "${TARGET_FREEZE}.sha256"
test -s "${TARGET_RECORD_HASHES}"
test -s "${TARGET_RECORD_HASHES}.sha256"
sha256sum --check "${TARGET_FREEZE}.sha256"
sha256sum --check "${TARGET_RECORD_HASHES}.sha256"

REPO_ROOT="${REPO_ROOT}" "${VLLM_VENV}/bin/python" \
  "${REPO_ROOT}/examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py"

NEMO_RL_VLLM_PY_EXECUTABLE="${VLLM_VENV}/bin/python" \
PYTHONPATH="${REPO_ROOT}" \
  /opt/nemo_rl_venv/bin/python - <<'PY'
import ray


@ray.remote
def probe_actor_environment() -> dict[str, str]:
    import importlib.metadata as metadata
    import sys

    import vllm

    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "vllm": metadata.version("vllm"),
        "vllm_source": vllm.__file__,
    }


ray.init(num_cpus=2, include_dashboard=False)
try:
    actor_python = "/opt/nemo_vllm_venv/bin/python"
    result = ray.get(
        probe_actor_environment.options(
            runtime_env={
                "py_executable": actor_python,
                "env_vars": {
                    "PYTHONPATH": "/opt/nemo-rl",
                    "VIRTUAL_ENV": "/opt/nemo_vllm_venv",
                    "UV_PROJECT_ENVIRONMENT": "/opt/nemo_vllm_venv",
                },
            }
        ).remote()
    )
finally:
    ray.shutdown()

assert result["python"] == "3.13", result
assert result["vllm"] == "0.23.0", result
assert result["prefix"] == "/opt/nemo_vllm_venv", result
assert not result["vllm_source"].startswith("/opt/dynamo_venv"), result
print("Ray actor environment", result)
print("Ray actor startup: validated")
PY

echo 'Baked regular-vLLM HSG image validation passed.'
