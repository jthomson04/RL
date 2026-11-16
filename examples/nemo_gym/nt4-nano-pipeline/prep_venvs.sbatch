#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -J nt4nano-venv-overlay
# Builds the venvs overlay.sh mounts over the rl-gym image's (NRL_VENV_OVERLAY_DIR). Run once per cluster, from the
# root of this NeMo-RL checkout with its Megatron-Bridge and Gym submodules initialized:
#
#   NRL_VENV_OVERLAY_DIR=/lustre/.../nt4nano-dynamo-venvs \
#     sbatch -A <account> -p batch examples/nemo_gym/nt4-nano-pipeline/prep_venvs.sbatch
#
# The image's venvs come from the fork's lock. These are writable copies of the six a run uses, with the
# infrastructure packages this NeMo-RL needs installed --no-deps (Ray 2.58, TransferQueue 05d4cc3, nemo-lens
# 4d045c0 and their new dependencies), plus ai-dynamo 1.7.0.dev20261106 in the vLLM worker venv. Model numerics
# (the NM4 vLLM 0.29.0 build, mamba-ssm, TE, flash-attn, CUDA libraries) stay as the image built them. The venvs
# NT4 Nano jobs 8866324 (Dynamo) and 8948499 (native) ran with were built this way. etcd and NATS go in bin/.
#
# Each venv gets a .prepared marker only after every import check passes; overlay.sh refuses to start without it.
set -euo pipefail

NRL_VENV_OVERLAY_DIR="${NRL_VENV_OVERLAY_DIR:?NRL_VENV_OVERLAY_DIR is required (shared storage, ~60 GB)}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra/images/high_stripe/rl-gym.71219647.sqsh}"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ETCD_VERSION=3.5.21
NATS_VERSION=2.11.6

D="${NRL_VENV_OVERLAY_DIR}"
C=/opt/nemo-rl
RV=/opt/ray_venvs
VLLM_PATH="${RV}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
log() { echo "$(date +%T) $*"; }

if [[ ! -f "${NEMO_RL_ROOT}/3rdparty/Gym-workspace/Gym/pyproject.toml" ]]; then
  echo "ERROR: submit from a NeMo-RL checkout with the Gym submodule initialized (NEMO_RL_ROOT=${NEMO_RL_ROOT})" >&2
  exit 1
fi
mkdir -p "${D}/bin"
export UV_CACHE_DIR="${D}/.uv-cache"

# name:path in the image
VENVS="driver:/opt/nemo_rl_venv
vllm:${VLLM_PATH}
gym:${RV}/nemo_rl.environments.nemo_gym.NemoGym
megatron:${RV}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
syncrollout:${RV}/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
reassembler:${RV}/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor"

copies=""
for e in ${VENVS}; do
  n=${e%%:*}
  p=${e#*:}
  rm -f "${D}/${n}/.prepared"
  [[ -f "${D}/${n}/.copied" ]] || copies+="rm -rf ${D}/${n} && mkdir -p ${D}/${n} && cp -a ${p}/. ${D}/${n}/ && touch ${D}/${n}/.copied & "
done
if [[ -n "${copies}" ]]; then
  log "copy the image's venvs"
  srun --container-image="${CONTAINER}" --container-mounts=/lustre:/lustre --no-container-mount-home bash -c "${copies} wait"
fi
MOUNTS="/lustre:/lustre"
for e in ${VENVS}; do
  n=${e%%:*}
  p=${e#*:}
  [[ -f "${D}/${n}/.copied" ]] || { log "COPY FAILED: ${n}"; exit 1; }
  MOUNTS+=",${D}/${n}:${p}"
done

log "vLLM venv: ai-dynamo and the pins the Dynamo lock moves"
srun --container-image="${CONTAINER}" --container-mounts="${MOUNTS}" --no-container-mount-home bash -c "
  set -euo pipefail
  cd /tmp
  unset UV_OVERRIDE UV_CONSTRAINT UV_BUILD_CONSTRAINT UV_PROJECT UV_CONFIG_FILE
  P=${VLLM_PATH}/bin/python
  uv pip install --no-config --python \$P --no-deps ai-dynamo==1.7.0.dev20261106 ai-dynamo-runtime==1.7.0.dev20261106 \
    certifi==2026.10.23 pydantic==2.13.5 pydantic-core==2.46.5 redis==8.1.0
  \$P -c 'import importlib.metadata as m, dynamo.common.backend, dynamo.llm, vllm; from dynamo.llm import PythonAsyncEngine, EntrypointArgs; print(\"ai-dynamo\", m.version(\"ai-dynamo\"), \"vllm\", vllm.__version__)'
  \$P -c 'from vllm.model_executor.models.registry import ModelRegistry as R; a = [x for x in R.get_supported_archs() if \"emotron4\" in x]; assert a, \"no NM4 architecture in this vLLM\"; print(\"NM4 archs:\", sorted(a))'
  \$P -m dynamo.frontend --help | grep -q -- --router-session-affinity-max-entries
"

PKGS="ray==2.58.0 git+https://github.com/Ascend/TransferQueue.git@05d4cc3377bee9b0f2897bbb90914ee20c0aea39 git+https://github.com/NVIDIA-NeMo/Lens.git@4d045c03186b90afdc815db46b671f6a21bc7e98 opentelemetry-api==1.44.0 opentelemetry-sdk==1.44.0 opentelemetry-instrumentation==0.65b0 opentelemetry-semantic-conventions==0.65b0 opentelemetry-util-http==0.65b0 opentelemetry-instrumentation-asgi==0.65b0 opentelemetry-instrumentation-fastapi==0.65b0 asgiref==3.13.0 wrapt==2.4.1 deprecated==3.0.0 limits==5.8.0 slowapi==0.1.10 redis==8.1.0 certifi==2026.10.23 aiohttp==3.14.5 aiobotocore==3.10.0 botocore==1.43.115 boto3==1.43.115 s3transfer==0.19.3 jmespath==1.1.0 aioitertools==0.13.0 pyjwt==2.15.1 msgspec==0.21.1 multi-storage-client==0.51.0 bracex==3.0.1 wcmatch==10.2.1 xattr==1.3.0 tzdata==2025.3"
log "all six venvs: Ray, TransferQueue, nemo-lens and their dependencies"
srun --container-image="${CONTAINER}" --container-mounts="${MOUNTS}" --no-container-mount-home bash -c "
  set -euo pipefail
  # uv must not pick up the image's /opt/nemo-rl pyproject: its override-dependencies pin nemo-lens to the fork's rev.
  cd /tmp
  unset UV_OVERRIDE UV_CONSTRAINT UV_BUILD_CONSTRAINT UV_PROJECT UV_CONFIG_FILE
  for py in /opt/nemo_rl_venv/bin/python ${RV}/*/bin/python; do
    case \$py in
      */nemo_rl_venv/*|*NemoGym*|*megatron_policy_worker.MegatronPolicyWorker*|*SyncRolloutActor*|*RolloutReassemblerActor*|*VllmAsyncGenerationWorker*) ;;
      *) continue ;;
    esac
    echo \"=== \$py\"
    # nemo-lens keeps its 0.2.0 base version across revs, so uv would otherwise keep the fork's build.
    uv pip install --no-config --python \$py --no-deps --reinstall-package nemo-lens ${PKGS}
    # lark: the Dynamo lock has 1.3.1, except in vLLM venvs, where vLLM pins 1.2.2 (already installed).
    case \$py in */nemo_rl_venv/*|*NemoGym*|*RolloutReassemblerActor*) uv pip install --no-config --python \$py --no-deps lark==1.3.1 ;; esac
    \$py -c 'import ray, transfer_queue; from nemo.lens import span_attributes; assert ray.__version__ == \"2.58.0\", ray.__version__'
  done
"

log "etcd ${ETCD_VERSION} and NATS ${NATS_VERSION} (linux-arm64) for Dynamo's control plane"
_tmp="$(mktemp -d)"
curl -fsSL "https://github.com/etcd-io/etcd/releases/download/v${ETCD_VERSION}/etcd-v${ETCD_VERSION}-linux-arm64.tar.gz" | tar xz -C "${_tmp}"
curl -fsSL "https://github.com/nats-io/nats-server/releases/download/v${NATS_VERSION}/nats-server-v${NATS_VERSION}-linux-arm64.tar.gz" | tar xz -C "${_tmp}"
install -m 755 "${_tmp}/etcd-v${ETCD_VERSION}-linux-arm64/etcd" "${_tmp}/nats-server-v${NATS_VERSION}-linux-arm64/nats-server" "${D}/bin/"
rm -rf "${_tmp}"
"${D}/bin/etcd" --version | head -1
"${D}/bin/nats-server" --version

log "import checks with the Dynamo pin mounted"
N="${NEMO_RL_ROOT}"
MOUNTS+=",${N}/nemo_rl:${C}/nemo_rl,${N}/examples:${C}/examples"
MOUNTS+=",${N}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:${C}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MOUNTS+=",${N}/3rdparty/Gym-workspace/Gym:${C}/3rdparty/Gym-workspace/Gym"
srun --container-image="${CONTAINER}" --container-mounts="${MOUNTS}" --no-container-mount-home bash -c "
  set -euo pipefail
  cd ${C}
  check() { echo \"=== \$1\"; timeout 900 \"\$2\" -c \"\$3\"; }
  check megatron ${RV}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python 'import megatron.core, megatron.bridge, nemo_rl.models.megatron.setup, nemo_rl.models.megatron.nemotron4, nemo_rl.models.policy.workers.megatron_policy_worker'
  check vllm ${VLLM_PATH}/bin/python 'import nemo_rl.models.generation.vllm.vllm_worker_async, nemo_rl.models.generation.dynamo.engine_adapter, nemo_rl.models.generation.dynamo.vllm.adapter, nemo_rl.models.generation.dynamo.vllm.chat_engine, nemo_rl.models.generation.dynamo.vllm.frontend'
  check gym ${RV}/nemo_rl.environments.nemo_gym.NemoGym/bin/python 'import nemo_rl.environments.nemo_gym'
  check syncrollout ${RV}/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor/bin/python 'import nemo_rl.experience.sync_rollout_actor'
  check reassembler ${RV}/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor/bin/python 'import nemo_rl.experience.rollout_reassembler_actor'
  check driver /opt/nemo_rl_venv/bin/python 'import nemo_rl.algorithms.single_controller, nemo_rl.algorithms.single_controller_utils.setup, nemo_rl.environments.nemo_gym, nemo_rl.models.generation.dynamo.managed_runtime, nemo_rl.models.huggingface.common'
"

for e in ${VENVS}; do touch "${D}/${e%%:*}/.prepared"; done
log "DONE: NRL_VENV_OVERLAY_DIR=${D} is ready"
