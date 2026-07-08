#!/usr/bin/env bash
# Materialize and freeze the regular NeMo-RL vLLM 0.23 actor environment.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}
PERSISTENT_CACHE=${PERSISTENT_CACHE:-${ROOT}/cache/nemotron_nano_v3_5_vllm023}
NEMO_RL_VENV_DIR=${VLLM_COMPARISON_VENV_DIR:-${PERSISTENT_CACHE}/venvs}
VLLM_ACTOR=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
VLLM_VENV=${NEMO_RL_VENV_DIR}/${VLLM_ACTOR}
UV_CACHE_DIR=${UV_CACHE_DIR:-${PERSISTENT_CACHE}/uv}
MANIFEST_DIR=${VLLM_ACTOR_MANIFEST_DIR:-${PERSISTENT_CACHE}/manifests}
FREEZE_FILE=${MANIFEST_DIR}/vllm023-actor-freeze.txt
FREEZE_SHA_FILE=${FREEZE_FILE}.sha256
STACK_MARKER=${VLLM_VENV}/.dynamo-vllm-stack
PINNED_REQUIREMENTS=${REPO_ROOT}/examples/swe_bench/vllm023_actor_freeze.txt
VALIDATOR=${REPO_ROOT}/examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py

cd "${REPO_ROOT}"
mkdir -p "${NEMO_RL_VENV_DIR}" "${UV_CACHE_DIR}" "${MANIFEST_DIR}"
rm -f "${STACK_MARKER}"

if [[ ! -e "${VLLM_VENV}/bin/python" ]]; then
  UV_PROJECT_ENVIRONMENT="${VLLM_VENV}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    uv sync --locked --extra vllm --directory "${REPO_ROOT}"
fi

if ! "${VLLM_VENV}/bin/python" -c \
  'import importlib.metadata as m; assert m.version("vllm") == "0.23.0"' \
  >/dev/null 2>&1; then
  if [[ -s "${PINNED_REQUIREMENTS}" ]]; then
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
      uv pip sync --python "${VLLM_VENV}/bin/python" "${PINNED_REQUIREMENTS}"
  else
    echo "Bootstrapping vLLM 0.23 before the first environment freeze." >&2
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
      uv pip install --python "${VLLM_VENV}/bin/python" --upgrade 'vllm==0.23.0'
  fi
fi

ACTOR_SITE=$(
  "${VLLM_VENV}/bin/python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
ACTOR_RELOAD_META=${ACTOR_SITE}/vllm/model_executor/model_loader/reload/meta.py
DYNAMO_RELOAD_META=/opt/dynamo_venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/reload/meta.py
if ! cmp -s "${ACTOR_RELOAD_META}" "${DYNAMO_RELOAD_META}"; then
  VLLM_RELOAD_PATCH=${REPO_ROOT}/docker/patches/vllm-0.23.0-layerwise-reload-composed-loader.patch
  test -f "${VLLM_RELOAD_PATCH}"
  (
    cd "${ACTOR_SITE}"
    git apply --check "${VLLM_RELOAD_PATCH}"
    git apply "${VLLM_RELOAD_PATCH}"
  )
fi
cmp -s "${ACTOR_RELOAD_META}" "${DYNAMO_RELOAD_META}"

REPO_ROOT="${REPO_ROOT}" "${VLLM_VENV}/bin/python" "${VALIDATOR}"

freeze_tmp=${FREEZE_FILE}.tmp.$$
sha_tmp=${FREEZE_SHA_FILE}.tmp.$$
marker_tmp=${STACK_MARKER}.tmp.$$
uv pip freeze --python "${VLLM_VENV}/bin/python" > "${freeze_tmp}"
freeze_sha=$(sha256sum "${freeze_tmp}" | awk '{print $1}')
printf '%s  %s\n' "${freeze_sha}" "$(basename "${FREEZE_FILE}")" > "${sha_tmp}"
printf '%s\n' \
  'vllm=0.23.0' \
  'python=3.13' \
  'refit=vllm#44814@45ffb397' \
  "freeze_sha256=${freeze_sha}" > "${marker_tmp}"
mv "${freeze_tmp}" "${FREEZE_FILE}"
mv "${sha_tmp}" "${FREEZE_SHA_FILE}"
mv "${marker_tmp}" "${STACK_MARKER}"

echo "Prepared regular-vLLM actor environment: ${VLLM_VENV}"
echo "Freeze: ${FREEZE_FILE}"
echo "Freeze SHA256: ${freeze_sha}"
