#!/usr/bin/env bash
# Bake the E2E-proven regular-vLLM actor environment into the combined image.

set -euo pipefail

ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}
REPO=${REPO:-${ROOT}/RL}
PERSISTENT_CACHE=${PERSISTENT_CACHE:-${ROOT}/cache/nemotron_nano_v3_5_vllm023}
VLLM_ACTOR=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
SOURCE_VENV=${SOURCE_VENV:-${PERSISTENT_CACHE}/venvs/${VLLM_ACTOR}}
SOURCE_MANIFEST_DIR=${SOURCE_MANIFEST_DIR:-${PERSISTENT_CACHE}/manifests}
SOURCE_FREEZE=${SOURCE_MANIFEST_DIR}/vllm023-actor-freeze.txt
SOURCE_FREEZE_SHA=${SOURCE_FREEZE}.sha256
TARGET_REPO=/opt/nemo-rl
TARGET_VENV=/opt/nemo_vllm_venv
TARGET_FREEZE=/opt/nemo_vllm_actor_freeze.txt
TARGET_RECORD_HASHES=/opt/nemo_vllm_actor_record_hashes.txt

test "$(uname -m)" = aarch64
test -x "${SOURCE_VENV}/bin/python"
test -s "${SOURCE_VENV}/.dynamo-vllm-stack"
test -s "${SOURCE_FREEZE}"
test -s "${SOURCE_FREEZE_SHA}"
(
  cd "${SOURCE_MANIFEST_DIR}"
  sha256sum --check "$(basename "${SOURCE_FREEZE_SHA}")"
)

# Revalidate the exact environment that passed E2E before copying it.
REPO_ROOT="${REPO}" "${SOURCE_VENV}/bin/python" \
  "${REPO}/examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py"

# Keep the image self-contained. The source actor environment is a relocatable
# uv venv, so copying preserves the exact wheels and compiled extensions used
# during E2E rather than resolving them again during image construction.
rsync -a --delete --exclude=.git --exclude=.venv "${REPO}/" "${TARGET_REPO}/"
rm -rf "${TARGET_VENV}"
cp -a "${SOURCE_VENV}" "${TARGET_VENV}"
test -x "${TARGET_VENV}/bin/python"

# Refresh only NeMo-RL's editable pointer; the copied environment originally
# points at the Lustre checkout. --no-deps guarantees that no runtime package
# can be added, removed, or upgraded during this step.
UV=$(command -v uv)
"${UV}" pip install \
  --python "${TARGET_VENV}/bin/python" \
  --no-deps \
  --editable "${TARGET_REPO}"

"${UV}" pip freeze --python "${TARGET_VENV}/bin/python" > "${TARGET_FREEZE}"
sha256sum "${TARGET_FREEZE}" > "${TARGET_FREEZE}.sha256"

# RECORD files contain the wheel-level content hashes. Hashing each RECORD
# gives the packaged image an auditable fingerprint in addition to its package
# freeze and source-environment checksum.
find "${TARGET_VENV}/lib/python3.13/site-packages" \
  -type f -path '*/RECORD' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${TARGET_RECORD_HASHES}"
sha256sum "${TARGET_RECORD_HASHES}" > "${TARGET_RECORD_HASHES}.sha256"
cp "${SOURCE_FREEZE}" /opt/nemo_vllm_actor_e2e_freeze.txt
cp "${SOURCE_FREEZE_SHA}" /opt/nemo_vllm_actor_e2e_freeze.txt.sha256

source_freeze_sha=$(sha256sum "${SOURCE_FREEZE}" | awk '{print $1}')
baked_freeze_sha=$(sha256sum "${TARGET_FREEZE}" | awk '{print $1}')
record_hashes_sha=$(sha256sum "${TARGET_RECORD_HASHES}" | awk '{print $1}')
cat > "${TARGET_VENV}/.dynamo-vllm-stack" <<EOF
vllm=0.23.0
python=3.13
refit=vllm#44814@45ffb397
e2e_freeze_sha256=${source_freeze_sha}
baked_freeze_sha256=${baked_freeze_sha}
record_hashes_sha256=${record_hashes_sha}
EOF

REPO_ROOT="${TARGET_REPO}" "${TARGET_VENV}/bin/python" \
  "${TARGET_REPO}/examples/swe_bench/validate_nano_v3_5_swe_vllm_hsg.py"

echo "Baked regular-vLLM actor environment: ${TARGET_VENV}"
echo "Baked freeze SHA256: ${baked_freeze_sha}"
echo "Package RECORD manifest SHA256: ${record_hashes_sha}"
