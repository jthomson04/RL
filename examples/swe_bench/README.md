# SWE-bench Dynamo rollout experiments

End-to-end SWE-bench Verified rollout configs that target K8s-hosted Dynamo
serving on the GB300 customer-cpu fleet. Each subfolder under this directory
is a self-contained experiment that any same-namespace operator can run with
one command.

## Folder convention

```
examples/swe_bench/
├── README.md                     ← this file
├── run_test.sh                   ← shared driver wrapper
└── <experiment-name>/
    ├── recipe.yaml               ← GRPO recipe consumed by run_dynamo_rollout_only.py
    ├── infra.gb300.yaml          ← nrl-k8s infra: RayCluster + dynamo.serving → ./dgd.gb300.yaml
    └── dgd.gb300.yaml            ← Dynamo DGD manifest (vLLM worker count, GPU, TP)
```

`run_test.sh` resolves an experiment name to `<exp>/recipe.yaml` +
`<exp>/infra.gb300.yaml` automatically. The infra YAML in turn references
its sibling `dgd.gb300.yaml` via a relative path, so the three files form a
co-located, version-controlled triple.

**All production-relevant values** (dataset path, `step_limit`, `concurrency`,
`run_golden`, `num_generations_per_prompt`, vLLM worker count / TP) **are
baked into the YAMLs** — no Hydra overrides at the entrypoint. The only
runtime injections are the two per-user template values (`dgd_name`,
`frontend_port`) that cannot live inline in a shared recipe.

## Running an experiment

From the `RL/` repo root, after the prereqs below:

    ./examples/swe_bench/run_test.sh <experiment-name> [flags]

For example:

    ./examples/swe_bench/run_test.sh qwen3_30b_a3b_instruct_2507 --skip-reinstall --skip-checkout

Useful flags (run `--help` for the full list, which also dumps available
experiments):

| Flag | When |
| --- | --- |
| `--skip-reinstall` | nrl-k8s already points at this checkout |
| `--skip-checkout` | you're already on the right branch |
| `--skip-sync` | PVC checkout is already current |
| `--no-down` | keep RayCluster + DGD up after the run (debugging / offline replay) |
| `--down-only` | just tear down a leftover cluster, no submit |

Env overrides (set before the command, the script picks them up):

| Var | Default | When to override |
| --- | --- | --- |
| `BRANCH` | current branch | reproduce a specific branch state |
| `RECIPE` / `INFRA` | auto-resolved from experiment name | a folder ships >1 infra variant (e.g. different hardware) |
| `SYNC_POD` | `$USER-dev-pod` | named differently in your namespace |
| `SYNC_SCRIPT` | `/lustre/.../$USER/.../script/k8s/sync_to_pod.sh` | your sync helper lives elsewhere |
| `FOLLOW_TIMEOUT_S` | `7200` (2h) | tighter timeout for fast experiments / longer for slow ones |

## Prereqs (one-time per shell)

Each must succeed before `run_test.sh` will start.

1. **AWS SSO + kubectl context fresh** — token expires every ~15 min:

       source /lustre/fsw/portfolios/coreai/users/${USER}/evolution_rl/script/k8s/k8s_auth.sh
       kubectl auth can-i list dynamographdeployments.nvidia.com -n default

2. **Model snapshot in your HF cache** (the recipe pins the model name):

       kubectl exec -n default ${USER}-dev-pod -- bash -c \
         "ls /mnt/rl-workspace/${USER}/hf_home/hub/ 2>/dev/null | grep -i <model>"

   If missing:

       huggingface-cli download <model-id> --cache-dir /mnt/rl-workspace/${USER}/hf_home

3. **SWE-bench SIF container cache** — jthomson04 maintains the shared copy:

       kubectl exec -n default ${USER}-dev-pod -- bash -c \
         'ls /mnt/rl-workspace/jothomson/swebench_containers/swebench_sweb.eval.arm64.*.sif | wc -l'
       # expect 281 for Verified

4. **Dataset row count matches the recipe**:

       kubectl exec -n default ${USER}-dev-pod -- bash -c \
         'wc -l /mnt/rl-workspace/jothomson/swebench_containers/swebench_verified_arm64_mini_swe.jsonl'
       # expect 281

5. **KAI scheduler queue has enough GPU headroom**. Each experiment's
   `dgd.gb300.yaml` says how many GPUs it needs (`replicas` × `gpu`).
   Check your queue:

       kubectl get queue rl-${USER} -o yaml | yq '.spec.resources'

   If quota is short, set `nvidia.com/kai-scheduler-queue: backfill` in
   the experiment's `infra.gb300.yaml → dynamo.serving.annotations` for
   one-off backfill scheduling.

6. **No leftover smoke RayCluster** under the same name. The smoke
   baseline (`infra/nrl_k8s/examples/grpo_mini_swe_qwen3_30b_a3b_instruct_2507.rollout.gb300.infra.yaml`)
   uses the same RayCluster name on purpose so `run_test.sh`'s head-pod
   detection works for both — but they can't coexist:

       nrl-k8s cluster down \
         examples/nemo_gym/grpo_mini_swe_qwen3_30b_a3b_instruct_2507.yaml \
         --infra infra/nrl_k8s/examples/grpo_mini_swe_qwen3_30b_a3b_instruct_2507.rollout.gb300.infra.yaml \
         --wait

## Verifying a finished run

Replace `<exp-name>` with the experiment folder and `<exp-log-dir>` with the
recipe's `logger.log_dir` (each experiment's `recipe.yaml` declares its own
log dir, e.g. `logs/grpo-mini-swe-qwen3-30b-a3b-instruct-2507-rollout-full`).

1. **Driver exit_code = 0**. `run_test.sh` already polls the head pod's
   `/tmp/nrl-${RUN_ID}/exitcode` file and exits non-zero if it isn't `0`.
   For a manual check after `--no-down`:

       HEAD_POD=$(kubectl get pod -n default \
         -l "ray.io/cluster=${USER}-rc-mini-swe-qwen3-30b-a3b,ray.io/node-type=head" \
         -o jsonpath='{.items[0].metadata.name}')
       kubectl exec -n default $HEAD_POD -- cat /tmp/nrl-<RUN_ID>/exitcode

2. **Trajectory rows match dataset size**:

       kubectl exec -n default ${USER}-dev-pod -- bash -c \
         "wc -l /mnt/rl-workspace/${USER}/nemo-rl/<exp-log-dir>/exp_*/trajectory_collection.jsonl" \
         | tail -1
       # expect the full row count (281 for Verified)

3. **Prometheus monitor bundle complete**:

       EXP=/mnt/rl-workspace/${USER}/nemo-rl/<exp-log-dir>/exp_NNN/dynamo_prometheus_export
       kubectl exec -n default ${USER}-dev-pod -- python3 -c "
       import json
       c={}
       for l in open('$EXP/raw_scrapes.jsonl'):
           ep = json.loads(l).get('endpoint_name','?')
           c[ep] = c.get(ep, 0) + 1
       print(c)"
       # expect 3 endpoints scraping consistently: vllmdecodeworker / frontend / dcgm

       # Cross-check the DCGM GPU label count vs the DGD's gpus_total:
       kubectl exec -n default ${USER}-dev-pod -- bash -c \
         "grep '^DCGM_FI_DEV_GPU_UTIL{' $EXP/data.openmetrics | grep -oE 'gpu=\"[0-9]+\"' | sort -u | wc -l"

4. **Offline Grafana replay** (on Mac):

       view-dynamo-monitoring-offline.sh /path/to/local/exp_NNN

   All 48 dashboard panels should have data — production workload drives
   the histogram_quantile / rate windows that the smoke leaves empty.

## Adding a new experiment

1. Pick a folder name. The script matches by exact folder name; convention
   is `<model>_<scale>_<variant>`, e.g. `qwen3_8b_thinking` or
   `gpt_oss_120b_high_throughput`.

2. Copy an existing experiment as a starting point:

       cp -r examples/swe_bench/qwen3_30b_a3b_instruct_2507 \
             examples/swe_bench/<new-experiment>

3. Edit the three YAMLs:
   - `recipe.yaml`: `policy.model_name`, `mini_swe_agent.step_limit` /
     `.concurrency`, `data.train.data_path` if you're targeting a
     non-Verified subset, `logger.log_dir`.
   - `dgd.gb300.yaml`: `services.VllmDecodeWorker.replicas`,
     `resources.requests.gpu`, `args:` to flip `--tensor-parallel-size`,
     `--max-model-len`, `--model`, etc.
   - `infra.gb300.yaml`: usually only `RESULTS_PARENT` in the entrypoint
     needs updating to match the new `logger.log_dir`. Everything else
     (RayCluster, dynamo.serving.name, HF_HOME, KAI queue) is
     `${user:}`-templated already.

4. Run via `./examples/swe_bench/run_test.sh <new-experiment>`.

## Notes / known risks

- **TP > 1 first-run hazard.** Each new DGD manifest that bumps tensor
  parallelism past 1 should be validated on the actual cluster, not just
  `nrl-k8s check`. Watch the first ~5 min of vLLM worker init — NCCL /
  NVLink failures surface there.
- **mini-swe-agent concurrency hang.** The agent runs as a single Ray
  actor with asyncio internal concurrency. If one sandbox subprocess
  hangs, it can stall the whole batch. If the driver log gets stuck on
  `Collecting rollouts: X/N` for >5 min, drop `concurrency` in the
  experiment's `recipe.yaml`.
- **PVC space.** The Prometheus export bundle scales with run length:
  281 samples × 56 min ≈ 60–100 MB `raw_scrapes.jsonl` per exp. Old
  `exp_NNN/` dirs accumulate forever unless you GC them.

## Provenance / what to point at

- Smoke baseline (1 sample, 1 GPU, TP=1) untouched at:
  - `examples/nemo_gym/grpo_mini_swe_qwen3_30b_a3b_instruct_2507.yaml`
  - `infra/nrl_k8s/examples/grpo_mini_swe_qwen3_30b_a3b_instruct_2507.rollout.gb300.infra.yaml`
  - `infra/nrl_k8s/examples_dgd/qwen3_30b_a3b_instruct_2507_gb300.yaml`
- Production sizing reference: jthomson04 2026-05-19 SWE-bench Verified
  run on Qwen3-30B-A3B-Instruct-2507 (4 worker × 2 GPU × TP=2,
  56 min for 281 samples, 4.99 rows/min).
- Dynamo Prometheus monitoring stack: branch `ruit/joyang/dynamo_rollout_poc`
  commit `2694a8900` (DCGM + frontend + bucket scrape + dashboard fidelity).
