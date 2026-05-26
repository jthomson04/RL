#!/usr/bin/env bash
# SWE-bench Dynamo rollout driver — one-button reproduction wrapper.
#
# Usage:
#   ./examples/swe_bench/run_test.sh <experiment-name> [flags]
#
# <experiment-name> is the subfolder name under examples/swe_bench/ that
# holds the recipe + infra + DGD for this experiment, e.g.
#   ./examples/swe_bench/run_test.sh qwen3_30b_a3b_instruct_2507
# which auto-resolves to
#   RECIPE=examples/swe_bench/qwen3_30b_a3b_instruct_2507/recipe.yaml
#   INFRA=examples/swe_bench/qwen3_30b_a3b_instruct_2507/infra.gb300.yaml
# Each experiment folder must contain at minimum `recipe.yaml` +
# `infra.gb300.yaml`; the infra YAML internally references its sibling
# `dgd.gb300.yaml`.
#
# Prereqs (one-time per shell — script bails early if missing):
#   1. uv and `nrl-k8s` itself reachable. The script will
#      `uv tool install --editable infra/nrl_k8s --reinstall` for you if
#      not already pointing at this checkout.
#   2. AWS SSO + kubectl context: `source <path>/k8s_auth.sh` BEFORE running.
#   3. **Sync the checkout to PVC yourself before running** — this script no
#      longer rsyncs. A typical incantation:
#        bash <path>/script/k8s/sync_to_pod.sh <path>/RL /mnt/rl-workspace/$USER/nemo-rl/
#      Run this whenever you've edited the recipe/infra/code and want the
#      pods to see the change. Skipping the rsync from inside this script
#      keeps the PVC under deliberate operator control (the previous
#      auto-sync silently produced PVC cruft when paths were misconfigured).
#
# What this script does:
#   1. (idempotent) Reinstall the local `nrl-k8s` editable build.
#   2. (optional) `git checkout` the configured branch.
#   3. `nrl-k8s check` (validate recipe + infra render).
#   4. `nrl-k8s run --raycluster --no-wait` (apply RayCluster + DGD, nohup
#      the driver on head pod).
#   5. Tail the driver log on PVC until the `${RUN_ID}` exitcode marker shows up.
#   6. `nrl-k8s cluster down --wait` (tear down RayCluster + DGD).
#
# Flags / env overrides:
#   --skip-reinstall      Skip the editable reinstall of nrl-k8s.
#   --skip-checkout       Skip `git checkout` (use whatever branch is current).
#   --no-follow           Submit + return immediately (don't tail driver log).
#   --no-down             Keep RayCluster + DGD up after run (for debugging).
#   --down-only           Just tear down; no submit.
#
#   BRANCH                Branch to checkout (default: current branch).
#   RECIPE / INFRA        Override the auto-resolved recipe / infra paths
#                         (e.g. when an experiment folder ships multiple
#                         infra variants for different hardware).
#   SYNC_POD              Pod label used in the "did you sync" reminder
#                         (default: $USER-dev-pod; informational only).
#   FOLLOW_TIMEOUT_S      Driver-watch timeout in seconds (default: 7200 = 2h).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR = .../RL/examples/swe_bench/  →  RL_ROOT = .../RL/  (two levels up)
RL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BRANCH="${BRANCH:-$(git -C "${RL_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")}"
# SYNC_POD is informational here — this script no longer rsyncs. It's still
# read so log lines reference the pod used for the polled exitcode/driver-log
# (same convention as the standalone sync helper).
SYNC_POD="${SYNC_POD:-${USER}-dev-pod}"
NRL_K8S_SRC="${RL_ROOT}/infra/nrl_k8s/src/nrl_k8s"
FOLLOW_TIMEOUT_S="${FOLLOW_TIMEOUT_S:-7200}"

SKIP_REINSTALL=false
SKIP_CHECKOUT=false
FOLLOW=true
TEARDOWN=true
DOWN_ONLY=false
EXPERIMENT=""

log() { printf '==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }
require_cmd() { need_cmd "$1" || die "$1 is required on PATH"; }

print_help_and_list_experiments() {
  sed -n '2,/^set -/p' "$0" | sed -e 's/^# \{0,1\}//' -e '/^set -/d'
  echo ""
  echo "Available experiments:"
  if [ -d "${SCRIPT_DIR}" ]; then
    for d in "${SCRIPT_DIR}"/*/; do
      [ -d "$d" ] || continue
      name=$(basename "$d")
      if [ -f "$d/recipe.yaml" ] && [ -f "$d/infra.gb300.yaml" ]; then
        echo "  - ${name}"
      fi
    done
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-reinstall) SKIP_REINSTALL=true; shift ;;
    --skip-checkout)  SKIP_CHECKOUT=true; shift ;;
    --no-follow)      FOLLOW=false; shift ;;
    --no-down)        TEARDOWN=false; shift ;;
    --down-only)      DOWN_ONLY=true; shift ;;
    -h|--help)
      print_help_and_list_experiments
      exit 0
      ;;
    --*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)
      if [ -z "$EXPERIMENT" ]; then
        EXPERIMENT="$1"; shift
      else
        echo "unexpected positional arg: $1 (already have experiment=${EXPERIMENT})" >&2
        exit 2
      fi
      ;;
  esac
done

# Resolve experiment folder into RECIPE + INFRA, unless the caller has
# explicitly overridden them on the env (escape hatch for multi-infra
# experiments where the folder ships >1 infra YAML variant).
if [ -z "$EXPERIMENT" ] && [ -z "${RECIPE:-}" ]; then
  echo "ERROR: missing experiment name." >&2
  echo ""
  print_help_and_list_experiments >&2
  exit 2
fi

if [ -n "$EXPERIMENT" ]; then
  EXP_DIR="${SCRIPT_DIR}/${EXPERIMENT}"
  [ -d "$EXP_DIR" ] || die "experiment folder not found: ${EXP_DIR}"
  RECIPE="${RECIPE:-${EXP_DIR}/recipe.yaml}"
  INFRA="${INFRA:-${EXP_DIR}/infra.gb300.yaml}"
fi
[ -f "$RECIPE" ] || die "recipe not found: ${RECIPE}"
[ -f "$INFRA" ]  || die "infra not found: ${INFRA}"

require_cmd git
require_cmd kubectl
require_cmd awk

cd "${RL_ROOT}"

# ---- early teardown-only path ----
if [ "${DOWN_ONLY}" = true ]; then
  require_cmd nrl-k8s
  log "cluster down only"
  nrl-k8s cluster down "${RECIPE}" --infra "${INFRA}" --wait
  exit 0
fi

# ---- auth sanity ----
if ! kubectl auth can-i list dynamographdeployments.nvidia.com -n default >/dev/null 2>&1; then
  die "kubectl can't reach the cluster — \`source k8s_auth.sh\` in this shell first"
fi

# ---- step 1: editable reinstall (idempotent) ----
nrl_k8s_matches_checkout() {
  if ! need_cmd nrl-k8s; then return 1; fi
  local launcher python_bin
  launcher="$(command -v nrl-k8s)"
  python_bin="$(awk 'NR == 1 { sub(/^#!/, ""); print; exit }' "${launcher}")"
  [ -n "${python_bin}" ] && [ -x "${python_bin}" ] || return 1
  "${python_bin}" - "${NRL_K8S_SRC}" <<'PY'
import pathlib, sys
expected = pathlib.Path(sys.argv[1]).resolve()
try:
    import nrl_k8s
except Exception:
    raise SystemExit(1)
actual = pathlib.Path(nrl_k8s.__file__).resolve().parent
raise SystemExit(0 if actual == expected else 1)
PY
}

if [ "${SKIP_REINSTALL}" = true ]; then
  log "skipping nrl-k8s reinstall (--skip-reinstall)"
elif nrl_k8s_matches_checkout; then
  log "nrl-k8s already points at this checkout — skipping reinstall"
else
  require_cmd uv
  log "reinstalling nrl-k8s editable from ${NRL_K8S_SRC%/src/nrl_k8s}"
  uv tool install --editable "${NRL_K8S_SRC%/src/nrl_k8s}" --reinstall
  hash -r 2>/dev/null || true
fi
require_cmd nrl-k8s

# ---- step 2: branch ----
if [ "${SKIP_CHECKOUT}" = true ] || [ -z "${BRANCH}" ]; then
  log "skipping git checkout (--skip-checkout or no BRANCH set); on $(git rev-parse --abbrev-ref HEAD)"
else
  log "git checkout ${BRANCH}"
  git checkout "${BRANCH}"
fi

# ---- step 3: preflight validate ----
# PVC rsync is intentionally NOT done here — sync your local checkout to
# /mnt/rl-workspace/$USER/nemo-rl/ manually before invoking this script
# (see header). The pod-side `nrl-k8s check` reads the *local* recipe/infra,
# so this check still catches local YAML breakage even without a sync.
log "nrl-k8s check"
nrl-k8s check "${RECIPE}" --infra "${INFRA}"

# ---- step 4: submit + (optional) tail driver log + (optional) teardown ----
SUBMIT_EPOCH=$(date +%s)

log "nrl-k8s run --raycluster --no-wait"
RUN_OUTPUT="$(nrl-k8s run "${RECIPE}" --infra "${INFRA}" --raycluster --no-wait 2>&1 | tee /dev/stderr)"
RUN_ID="$(awk '/^run id:/ {print $3; exit}' <<<"${RUN_OUTPUT}")"
[ -n "${RUN_ID}" ] || die "could not parse run id from nrl-k8s output"
log "run id: ${RUN_ID}"

# Cleanup trap: tear down on script exit unless --no-down.
cleanup_done=false
do_teardown() {
  ${cleanup_done} && return 0
  cleanup_done=true
  if [ "${TEARDOWN}" = true ]; then
    log "tearing down RayCluster + DGD"
    nrl-k8s cluster down "${RECIPE}" --infra "${INFRA}" --wait || warn "cluster down returned non-zero"
  else
    log "leaving RayCluster + DGD up (--no-down). Manual teardown:"
    echo "  nrl-k8s cluster down ${RECIPE} --infra ${INFRA} --wait"
  fi
}
trap do_teardown EXIT INT TERM

# ---- step 5: tail driver log until [result] markers appear ----
if [ "${FOLLOW}" = false ]; then
  log "submit complete; --no-follow set, returning without tailing"
  exit 0
fi

log "waiting for driver log to appear (timeout ${FOLLOW_TIMEOUT_S}s)..."
LOG=""
DEADLINE=$((SUBMIT_EPOCH + FOLLOW_TIMEOUT_S))
while [ -z "${LOG}" ] && [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  CAND=$(kubectl exec -n default "${SYNC_POD}" -- bash -c \
    "ls -t /mnt/rl-workspace/${USER}/driver_logs/*mini-swe-qwen3-30b-a3b-rollout-*.log 2>/dev/null | head -1" 2>/dev/null | tr -d '\r')
  if [ -n "${CAND}" ]; then
    MTIME=$(kubectl exec -n default "${SYNC_POD}" -- stat -c %Y "${CAND}" 2>/dev/null)
    if [ -n "${MTIME}" ] && [ "${MTIME}" -ge "${SUBMIT_EPOCH}" ]; then
      LOG="${CAND}"
    fi
  fi
  [ -z "${LOG}" ] && sleep 5
done
[ -n "${LOG}" ] || die "no fresh driver log appeared within ${FOLLOW_TIMEOUT_S}s"
log "tailing $(basename "${LOG}")"

# Background tail-follow + foreground poll for [result] / exit code.
kubectl exec -n default "${SYNC_POD}" -- tail -F "${LOG}" 2>/dev/null &
TAIL_PID=$!
# Make sure the tail dies when this script exits.
trap "kill ${TAIL_PID} 2>/dev/null || true; do_teardown" EXIT INT TERM

# Cluster name is the rendered value of ${user:}-rc-mini-swe-qwen3-30b-a3b
# from the infra YAML — `${user:}` resolves to $USER via OmegaConf.
CLUSTER_NAME="${USER}-rc-mini-swe-qwen3-30b-a3b"

while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  HEAD_POD=$(kubectl get pod -n default \
    -l "ray.io/cluster=${CLUSTER_NAME},ray.io/node-type=head" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [ -n "${HEAD_POD}" ]; then
    EXIT_CODE=$(kubectl exec -n default "${HEAD_POD}" -- bash -c \
      "cat /tmp/nrl-${RUN_ID}/exitcode 2>/dev/null || true" \
      2>/dev/null | tr -d '\r')
    if [ -n "${EXIT_CODE}" ]; then
      sleep 2  # let the background tail flush its last buffered lines
      log "driver finished, exit code: ${EXIT_CODE}"
      break
    fi
  fi
  sleep 10
done

kill ${TAIL_PID} 2>/dev/null || true

if [ -z "${EXIT_CODE:-}" ]; then
  warn "did not see driver exit within ${FOLLOW_TIMEOUT_S}s"
  exit 124
fi

if [ "${EXIT_CODE}" != "0" ]; then
  die "driver exited non-zero: ${EXIT_CODE}"
fi
log "run succeeded ✓"
