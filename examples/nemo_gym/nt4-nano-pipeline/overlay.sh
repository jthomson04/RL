# Sourced by ../nemotron-3.5-lightning/lightning35_launch.sh when NRL_VENV_OVERLAY_DIR is set: adapts the rl-gym
# image built for the NT4 fork (saumishr/RL e10075cd8) to this NeMo-RL. See README.md next to this file.
# Not executable on its own; it reads and sets the launcher's variables.

_ov="${NRL_VENV_OVERLAY_DIR}"
_ov_doc="${PROJECT_ROOT}/examples/nemo_gym/nt4-nano-pipeline/README.md"
for _ov_venv in driver vllm gym megatron syncrollout reassembler; do
  if [[ ! -f "${_ov}/${_ov_venv}/.prepared" ]]; then
    echo "ERROR: ${_ov}/${_ov_venv} is missing or incomplete; build it with prep_venvs.sbatch (${_ov_doc})" >&2
    exit 1
  fi
done
for _ov_bin in etcd nats-server; do
  if [[ ! -x "${_ov}/bin/${_ov_bin}" ]]; then
    echo "ERROR: ${_ov}/bin/${_ov_bin} is missing; rerun prep_venvs.sbatch (${_ov_doc})" >&2
    exit 1
  fi
done

# The image's venvs come from the fork's lock; these copies carry this NeMo-RL's Ray, TransferQueue and nemo-lens,
# plus ai-dynamo in the vLLM one. etcd and NATS are Dynamo's control plane. The image's examples/ is the fork's, so
# mount this checkout's for run_grpo_single_controller.py.
_ov_rv=/opt/ray_venvs
_ov_mounts=(
  "${_ov}/driver:/opt/nemo_rl_venv"
  "${_ov}/vllm:${_ov_rv}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
  "${_ov}/gym:${_ov_rv}/nemo_rl.environments.nemo_gym.NemoGym"
  "${_ov}/megatron:${_ov_rv}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
  "${_ov}/syncrollout:${_ov_rv}/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
  "${_ov}/reassembler:${_ov_rv}/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor"
  "${_ov}/bin:/opt/dynamo-bin"
  "${PROJECT_ROOT}/examples:/opt/nemo-rl/examples"
)
_ov_joined="$(
  IFS=,
  printf '%s' "${_ov_mounts[*]}"
)"
export EXTRA_MOUNTS="${_ov_joined}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"
export NEMO_RL_DYNAMO_BIN_DIR=/opt/dynamo-bin

# The image's Gym server venvs carry the fork's Ray; give them this NeMo-RL's in every Ray container before Ray starts.
_ov_setup="bash /opt/nemo-rl/examples/nemo_gym/nt4-nano-pipeline/gym_venvs_ray.sh"
if [[ -n "${EXTRA_SETUP_COMMAND:-}" ]]; then
  EXTRA_SETUP_COMMAND+=$'\n'"${_ov_setup}"
else
  EXTRA_SETUP_COMMAND="${_ov_setup}"
fi

# HSG's job reaper cancels jobs whose GPUs idle for 30 minutes. Ray, Megatron and vLLM loading, the judge pools and
# the first rollouts leave the training GPUs idle for 30-35 minutes (smoke 8788671 was reaped), so ask for 60.
_ov_reaper='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"model_loading","description":"NeMo RL GRPO: Ray, Megatron and vLLM model loading, judge servers and first rollouts take about 35 min before the training GPUs compute"}}'
SLURM_COMMENT="${SLURM_COMMENT:-${_ov_reaper}}"

# Gym runs from this checkout, not the image. Without ether0's data every ether0 server clones it at startup into the
# same shared checkout, and the replicas race; clone it once here.
_ov_ether0="${GYM_SOURCE:-${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym}/resources_servers/ether0/.ether0"
if [[ ! -d "${_ov_ether0}" ]]; then
  git clone --depth=1 https://github.com/Future-House/ether0 "${_ov_ether0}" || {
    echo "ERROR: could not clone ether0 into ${_ov_ether0}" >&2
    exit 1
  }
fi
echo "  NT4 venv overlay: ${_ov}"
unset _ov _ov_doc _ov_venv _ov_bin _ov_rv _ov_mounts _ov_joined _ov_setup _ov_reaper _ov_ether0
