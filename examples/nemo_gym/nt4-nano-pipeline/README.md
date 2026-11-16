# NT4 Nano pipeline recipe on this NeMo-RL

This branch (`jthomson04/nt4-nano-dynamo`) is the NeMo-RL that `post-training/pipeline`'s
`RLVR/nemotron-4-nano` recipe pins as `nemo_rl_dynamo` for its `DYNAMO=1` arm. It is
NeMo-RL's Dynamo stack (NVIDIA-NeMo/RL #4979, #5115, #5117, as of Nov 10) plus the NT4 fork's
changes (`saumishr/RL` `e10075cd8`), ported. That fork's own Dynamo integration predates
NeMo-RL's and has since been removed.

The recipe's job is to show the Dynamo switch: its Dynamo configs are its own configs plus
`policy.generation.dynamo: true` and a frontend count. Everything needed to run the recipe on
this code instead of the fork lives here, on this branch:

- **Config chain.** Levels 2 and 3 (`examples/configs/recipes/llm/grpo-nt4nano-math-*.v1.yaml`,
  `nt4nano-nm4-train-base.yaml`) resolve the way job 7766157's chain did on the fork: ViT frozen,
  flash attention, FA2 set through the first-class `flash_attention_version`. They also have the
  two checkpoint settings this SingleController needs (`metric_name: null`,
  `save_data_plane: true`). These apply with and without Dynamo.
- **`overlay.sh`.** The recipe's rl-gym image was built for the fork. When `NRL_VENV_OVERLAY_DIR` is
  set, the Lightning launcher sources this file, which:
  - mounts six venvs upgraded to this branch's lock, plus etcd and NATS;
  - mounts this checkout's `examples/`;
  - puts this branch's Ray into the image's Gym server venvs (`gym_venvs_ray.sh`);
  - clones ether0's data once;
  - asks HSG's idle-GPU reaper for 60 minutes.

  The recipe's `DYNAMO=1` sets `NRL_VENV_OVERLAY_DIR`.
- **`prep_venvs.sbatch`** builds those venvs, once per cluster. On OCI-HSG the recipe defaults to a
  prepared copy.
- **`tools/external_gym_vllm/vllm_pool_lb.py`** allows 7500 s per judge call. At 16K/lag 8 GenRM's
  backlog outlasted the 1800 s default, and every timeout also quarantined a backend.

The 16K/lag-8 jobs 8866324 (Dynamo) and 8948499 (native, `++policy.generation.dynamo=false`) ran
this code. The pipeline recipe's README lists the validated runs.
