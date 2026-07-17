# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Post-rollout analysis for SWE-bench Dynamo experiments.

Reads an `exp_NNN` directory produced by `run_dynamo_rollout_only.py` and
writes two artifacts next to the existing `trajectory_collection.jsonl`:

  * ``summary.txt``               — human-readable run report (resolved/total,
                                    per-repo, outcome partition, token & timing
                                    aggregates, latency percentiles, time
                                    breakdown computed from vllm prom snapshots).
  * ``tool_call_timings.jsonl``   — one row per LLM API call across all
                                    instances. Contains per-request token usage,
                                    measured ``llm_time_ms`` (vllm-side e2e
                                    latency), derived ``tool_time_s``
                                    (agent-side gap to the next call), and
                                    estimated ``est_prefill_ms`` /
                                    ``est_decode_ms`` columns derived from the
                                    run's global ITL.

Runs both as a library function (called from `run_dynamo_rollout_only.py`'s
archive step via ``from examples.swe_bench.analysis import write_summary``)
and standalone for offline re-aggregation::

    python examples/swe_bench/analysis.py /path/to/exp_NNN
    python examples/swe_bench/analysis.py /path/to/exp_NNN --model Qwen/Qwen3-30B-A3B-Instruct-2507

The CLI auto-detects the model from ``<exp>/manifests/recipe.yaml`` when
``--model`` is omitted; falls back to the ``model`` field on the first
LLM response in any ``.traj.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


# =========================================================================
# Tiny utilities
# =========================================================================

def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolated percentile from a pre-sorted list (0 ≤ p ≤ 100)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _autodetect_model_from_manifests(exp_dir: Path) -> str | None:
    """Read `<exp>/manifests/recipe.yaml` and pull `policy.model_name`."""
    recipe = exp_dir / "manifests" / "recipe.yaml"
    if not recipe.exists():
        return None
    # Avoid pulling YAML deps; the field is one line in the recipes we ship.
    pat = re.compile(r'^\s*model_name:\s*"?([^"\s#]+)"?\s*$', re.M)
    try:
        m = pat.search(recipe.read_text())
    except OSError:
        return None
    return m.group(1) if m else None


def _autodetect_model_from_trajectories(traj_root: Path) -> str | None:
    """Fallback: pull `responses[0].model` from any `.traj.json` and strip the
    `hosted_vllm/` prefix LiteLLM tacks on."""
    if not traj_root.exists():
        return None
    for traj_file in traj_root.rglob("*.traj.json"):
        try:
            data = json.loads(traj_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        responses = data.get("responses") or []
        for r in responses:
            if isinstance(r, dict):
                m = r.get("model")
                if isinstance(m, str) and m:
                    return m.removeprefix("hosted_vllm/")
        return None
    return None


# =========================================================================
# Per-instance outcome classification (reads .traj.json + report_*.json)
# =========================================================================

def _collect_per_instance_outcomes(
    traj_root: Path, model_name: str
) -> dict[str, dict[str, Any]]:
    """Walk `<traj_root>/results/verified/<model>/<instance>/` and build a map
    `instance_id -> outcome dict`. The outcome dict has the raw signals we
    use to classify the rollout — caller decides priority order."""
    out: dict[str, dict[str, Any]] = {}
    model_root = traj_root / "results" / "verified" / model_name
    if not model_root.exists():
        return out

    for inst_dir in sorted(model_root.iterdir()):
        if not inst_dir.is_dir():
            continue
        iid = inst_dir.name

        # ---- traj.json ----
        traj_files = sorted(inst_dir.glob("*.traj.json"))
        info = {}
        any_finish_length = False
        num_steps = 0
        if traj_files:
            try:
                traj = json.loads(traj_files[-1].read_text())
                info = traj.get("info") or {}
                responses = traj.get("responses") or []
                num_steps = len(responses)
                for resp in responses:
                    if not isinstance(resp, dict):
                        continue
                    for ch in resp.get("choices") or []:
                        if isinstance(ch, dict) and ch.get("finish_reason") == "length":
                            any_finish_length = True
                            break
                    if any_finish_length:
                        break
            except (OSError, json.JSONDecodeError):
                pass

        exit_status = info.get("exit_status", "<unknown>")
        submission = info.get("submission") or ""
        has_submission = bool(
            isinstance(submission, str) and submission.strip()
        )

        # ---- report_*.json (one per instance, latest wins) ----
        report_files = sorted(inst_dir.glob("report_*.json"))
        report_present = bool(report_files)
        resolved: bool | None = None
        patch_applied: bool | None = None
        if report_files:
            try:
                rd = json.loads(report_files[-1].read_text())
                # Report is keyed by instance_id (or whatever the eval used).
                inner = rd.get(iid) or next(iter(rd.values()), {})
                if isinstance(inner, dict):
                    resolved = bool(inner.get("resolved"))
                    psa = inner.get("patch_successfully_applied")
                    patch_applied = bool(psa) if psa is not None else None
            except (OSError, json.JSONDecodeError):
                pass

        out[iid] = {
            "exit_status": exit_status,
            "has_submission": has_submission,
            "report_present": report_present,
            "resolved": resolved,
            "patch_applied": patch_applied,
            "any_finish_length": any_finish_length,
            "num_steps": num_steps,
        }
    return out


def _classify_outcome(o: dict[str, Any]) -> str:
    """Apply the mutually-exclusive priority order documented in the plan."""
    if o["resolved"] is True:
        return "resolved"
    if not o["report_present"]:
        return "no_report"
    if o["patch_applied"] is False:
        return "patch_failed_apply"
    if not o["has_submission"]:
        return "empty_patch"
    return "wrong_patch"


# =========================================================================
# Per-request stats (token + timing per LLM API call)
# =========================================================================

def _collect_per_request_stats(
    traj_root: Path, model_name: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk `trajectories/results/verified/<model>/<instance>/*.traj.json` and
    return ``(per_request_records, steps_per_instance)``.

    Each per-request record captures the data we can read off one
    OpenAI-compatible response object: token usage (with cached prefill
    breakdown), vLLM ``nvext.timing.total_time_ms`` server-side latency, and
    a derived ``tool_time_s`` = the gap until the next request minus this
    request's LLM serve time (the time the agent spent running its bash
    command + parsing + composing the next prompt).
    """
    per_request: list[dict[str, Any]] = []
    steps_per_instance: dict[str, int] = {}
    model_root = traj_root / "results" / "verified" / model_name
    if not model_root.exists():
        return per_request, steps_per_instance

    for instance_dir in sorted(model_root.iterdir()):
        if not instance_dir.is_dir():
            continue
        traj_files = sorted(instance_dir.glob("*.traj.json"))
        if not traj_files:
            continue
        try:
            traj = json.loads(traj_files[-1].read_text())
        except (OSError, json.JSONDecodeError):
            continue
        responses = traj.get("responses") or []
        instance_id = instance_dir.name
        steps_per_instance[instance_id] = len(responses)
        for i, resp in enumerate(responses):
            if not isinstance(resp, dict):
                continue
            usage = resp.get("usage") or {}
            ptd = usage.get("prompt_tokens_details") or {}
            timing = (resp.get("nvext") or {}).get("timing") or {}
            created = int(resp.get("created") or 0)
            llm_ms = float(timing.get("total_time_ms") or 0.0)
            tool_time_s: float | None = None
            if i + 1 < len(responses) and isinstance(responses[i + 1], dict):
                next_created = int(responses[i + 1].get("created") or 0)
                if next_created and created:
                    tool_time_s = max(0.0, next_created - created - llm_ms / 1000.0)
            per_request.append(
                {
                    "instance_id": instance_id,
                    "step": i,
                    "created": created,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "cached_tokens": int(ptd.get("cached_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "llm_time_ms": llm_ms,
                    "tool_time_s": tool_time_s,
                }
            )
    return per_request, steps_per_instance


# =========================================================================
# vLLM Prometheus snapshot diff (before/after .prom files in exp_dir)
# =========================================================================

# Metrics we pull, summed across all label-sets (worker, engine, etc.).
_PROM_METRICS = (
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def _parse_prom_file(path: Path) -> dict[str, float]:
    """Sum each metric in `_PROM_METRICS` across all label combinations.

    Prom text format is line-based; we ignore `# HELP` / `# TYPE` lines and
    parse `<metric>{labels} <value> [<timestamp>]`. Tolerates missing files
    by returning an empty dict.
    """
    totals: dict[str, float] = {}
    if not path.exists():
        return totals
    try:
        text = path.read_text()
    except OSError:
        return text  # type: ignore[return-value]
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # metric{labels} value [ts]    OR    metric value [ts]
        # Extract metric name (everything before `{` or first space).
        brace = line.find("{")
        space = line.find(" ")
        if brace != -1 and (space == -1 or brace < space):
            name = line[:brace]
            rest = line[line.find("}") + 1 :].strip()
        else:
            name = line[:space] if space != -1 else line
            rest = line[space + 1 :].strip() if space != -1 else ""
        if name not in _PROM_METRICS:
            continue
        # rest is "<value>" or "<value> <ts>"
        val_str = rest.split()[0] if rest else "0"
        try:
            val = float(val_str)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


def _read_vllm_prom_diff(exp_dir: Path) -> dict[str, float]:
    """Find before/after vllm prom snapshots in `exp_dir` and return the
    after − before deltas. Empty dict when snapshots aren't present (e.g.
    older exp_NNN dirs, or the 5/19 archive that pre-dates the snapshot
    convention).
    """
    before_files = sorted(exp_dir.glob("qwen*-before-*.prom"))
    after_files = sorted(exp_dir.glob("qwen*-after-*.prom"))
    if not before_files or not after_files:
        return {}
    before = _parse_prom_file(before_files[-1])
    after = _parse_prom_file(after_files[-1])
    if not before or not after:
        return {}
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in _PROM_METRICS}


def _compute_est_itl_ms(prom_diff: dict[str, float]) -> float | None:
    """Average inter-token latency (ms) over this run, from prom diff. Used
    to estimate per-request decode time. Returns None if the diff lacks
    ITL data (e.g. no snapshots)."""
    if not prom_diff:
        return None
    sum_s = prom_diff.get("vllm:inter_token_latency_seconds_sum", 0.0)
    n = prom_diff.get("vllm:inter_token_latency_seconds_count", 0.0)
    if n <= 0:
        return None
    return (sum_s / n) * 1000.0


# =========================================================================
# Summary rendering
# =========================================================================

def _render_summary(
    exp_dir: Path,
    model_name: str,
    rows: list[dict[str, Any]],
    per_request: list[dict[str, Any]],
    steps_per_instance: dict[str, int],
    outcomes: dict[str, dict[str, Any]],
    prom_diff: dict[str, float],
    est_itl_ms: float | None,
) -> str:
    """Compose the human-readable summary.txt body.

    Tolerates ``rows`` being empty (no trajectory_collection.jsonl, e.g. a
    standalone mini_swe_agent archive being re-aggregated offline). In that
    case, falls back to ``outcomes`` for the resolved/total + per-repo
    figures so the top block stays informative.
    """
    if rows:
        # NeMo-RL-side row-per-instance file present — use it as source of truth
        total_instances = len(rows)
        correct = sum(1 for r in rows if (r.get("reward") or 0) > 0.5)
        by_repo: Counter[str] = Counter()
        by_repo_correct: Counter[str] = Counter()
        for r in rows:
            iid = str(r.get("instance_id", ""))
            repo = iid.split("__", 1)[0] if "__" in iid else "?"
            by_repo[repo] += 1
            if (r.get("reward") or 0) > 0.5:
                by_repo_correct[repo] += 1
    else:
        # Fall back to outcomes (mini_swe_agent's report.json's resolved flag)
        total_instances = len(outcomes)
        correct = sum(1 for o in outcomes.values() if o["resolved"] is True)
        by_repo = Counter()
        by_repo_correct = Counter()
        for iid, o in outcomes.items():
            repo = iid.split("__", 1)[0] if "__" in iid else "?"
            by_repo[repo] += 1
            if o["resolved"] is True:
                by_repo_correct[repo] += 1
    avg_reward = (correct / total_instances) if total_instances else 0.0

    lines: list[str] = [
        f"Run: {exp_dir.name}",
        f"Model: {model_name}",
        f"Total instances: {total_instances}",
        f"Resolved: {correct}/{total_instances} "
        f"({100 * avg_reward:.2f}%)  avg_reward: {avg_reward:.4f}",
        "",
        "By repo (resolved / total):",
    ]
    if by_repo:
        name_width = max(len(r) for r in by_repo) + 2
        for repo, n in sorted(by_repo.items()):
            c = by_repo_correct.get(repo, 0)
            pct = 100 * c / n if n else 0.0
            lines.append(f"  {repo:<{name_width}} {c} / {n}  ({pct:.1f}%)")

    # ---- Outcome partition + flags ----
    if outcomes:
        partition_order = (
            "resolved",
            "no_report",
            "patch_failed_apply",
            "empty_patch",
            "wrong_patch",
        )
        partition_counts: Counter[str] = Counter(
            _classify_outcome(o) for o in outcomes.values()
        )
        total_o = sum(partition_counts.values())
        lines += [
            "",
            f"--- Outcomes (mutually exclusive, sum = {total_o}) ---",
        ]
        for cat in partition_order:
            n = partition_counts.get(cat, 0)
            if n == 0 and cat not in ("resolved", "wrong_patch"):
                continue  # collapse zero-rows except resolved/wrong_patch
            pct = 100 * n / total_o if total_o else 0.0
            lines.append(f"  {cat:<20s} {n:>4} / {total_o}  ({pct:5.2f}%)")

        exit_status_counts: Counter[str] = Counter(
            o["exit_status"] for o in outcomes.values()
        )
        truncation_n = sum(1 for o in outcomes.values() if o["any_finish_length"])
        lines += [
            "",
            "--- Flags (overlap with above) ---",
        ]
        for es, n in exit_status_counts.most_common():
            pct = 100 * n / total_o if total_o else 0.0
            lines.append(f"  exit={es:<20s} {n:>4} / {total_o}  ({pct:5.2f}%)")
        pct = 100 * truncation_n / total_o if total_o else 0.0
        lines.append(
            f"  truncation_in_any_response  {truncation_n:>4} / {total_o}  ({pct:5.2f}%)"
        )

    # ---- Per-request token + agent step + timing aggregates ----
    if per_request:
        total_steps = len(per_request)
        total_prompt = sum(r["prompt_tokens"] for r in per_request)
        total_cached = sum(r["cached_tokens"] for r in per_request)
        total_uncached = max(0, total_prompt - total_cached)
        total_decode = sum(r["completion_tokens"] for r in per_request)
        total_tokens = total_prompt + total_decode

        latencies_ms = sorted(r["llm_time_ms"] for r in per_request)
        output_sorted = sorted(r["completion_tokens"] for r in per_request)
        cached_sorted = sorted(r["cached_tokens"] for r in per_request)
        steps_sorted = sorted(steps_per_instance.values())

        all_created = [r["created"] for r in per_request if r["created"]]
        wall_s = (max(all_created) - min(all_created)) if all_created else 0
        cum_llm_s = sum(r["llm_time_ms"] for r in per_request) / 1000.0

        per_instance_llm_s: dict[str, float] = {}
        for r in per_request:
            per_instance_llm_s[r["instance_id"]] = (
                per_instance_llm_s.get(r["instance_id"], 0.0)
                + r["llm_time_ms"] / 1000.0
            )
        instance_llm_values = list(per_instance_llm_s.values())

        SLOW_MS = 5000.0
        slow_count = sum(1 for t in latencies_ms if t > SLOW_MS)
        slow_excess_s = (
            sum(t - SLOW_MS for t in latencies_ms if t > SLOW_MS) / 1000.0
        )

        lines += [
            "",
            "--- Tokens ---",
            f"Total Agent Steps:                {total_steps:>14,}",
            f"Total prefill tokens (cached+un):  {total_prompt:>14,}",
            f"  Cached prefill tokens:           {total_cached:>14,}",
            f"  Uncached prefill tokens:         {total_uncached:>14,}",
            f"Total decode tokens (output):      {total_decode:>14,}",
            f"Total tokens:                      {total_tokens:>14,}",
        ]
        if total_tokens > 0:
            lines += [
                f"  Prefill / total:                 {100 * total_prompt / total_tokens:>13.2f}%",
                f"  Decode  / total:                 {100 * total_decode / total_tokens:>13.2f}%",
                f"  Uncached prefill / total:        {100 * total_uncached / total_tokens:>13.2f}%",
            ]
        if total_steps > 0:
            lines += [
                f"Avg prompt tokens  / request:      {total_prompt / total_steps:>14,.1f}",
                f"Avg cached tokens  / request:      {total_cached / total_steps:>14,.1f}",
                f"Avg output tokens  / Agent Step:   {total_decode / total_steps:>14,.1f}",
            ]
        lines += [
            f"output_tokens p50:                 {_percentile(output_sorted, 50):>14,.1f}",
            f"output_tokens p95:                 {_percentile(output_sorted, 95):>14,.1f}",
            f"cached_tokens p50:                 {_percentile(cached_sorted, 50):>14,.1f}",
            f"cached_tokens p95:                 {_percentile(cached_sorted, 95):>14,.1f}",
            "",
            "--- Agent steps per instance ---",
        ]
        if steps_sorted:
            avg_steps = sum(steps_sorted) / len(steps_sorted)
            lines += [
                f"avg:    {avg_steps:>10.2f}",
                f"median: {_percentile(steps_sorted, 50):>10.1f}",
                f"p95:    {_percentile(steps_sorted, 95):>10.1f}",
            ]

        lines += [
            "",
            "--- Timing ---",
            f"Wall-clock:                        {wall_s:>10,d} s  ({wall_s / 60:.1f} min)",
            f"Total Agent Steps:                 {total_steps:>10,}",
            f"Cumulative LLM time:               {cum_llm_s:>10,.1f} s  ({cum_llm_s / 60:.1f} min)",
        ]
        if instance_llm_values:
            lines += [
                f"Per-instance LLM time (avg):       {sum(instance_llm_values) / len(instance_llm_values):>10,.1f} s",
                f"Per-instance LLM time (max):       {max(instance_llm_values):>10,.1f} s",
            ]
        if latencies_ms:
            lines += [
                "",
                "--- Per-request LLM latency (ms) ---",
                f"min: {min(latencies_ms):>10,.1f}",
                f"p10: {_percentile(latencies_ms, 10):>10,.1f}",
                f"p50: {_percentile(latencies_ms, 50):>10,.1f}",
                f"p90: {_percentile(latencies_ms, 90):>10,.1f}",
                f"p95: {_percentile(latencies_ms, 95):>10,.1f}",
                f"p99: {_percentile(latencies_ms, 99):>10,.1f}",
                f"max: {max(latencies_ms):>10,.1f}",
                f"avg: {sum(latencies_ms) / len(latencies_ms):>10,.1f}",
                "",
                f"Slow-tail requests (>5 s):         {slow_count:>6} of {total_steps}  "
                f"({100 * slow_count / total_steps:.2f}%)",
                f"Slow-tail excess time (Σ(t−5s)):   {slow_excess_s:>10,.1f} s",
            ]
    else:
        lines += [
            "",
            "(no per-request stats — trajectories/ tree empty or unreadable)",
        ]

    # ---- Time breakdown from vllm prom diff (server-side measured) ----
    # Independent of per_request: prom snapshots are exp_dir/*.prom and exist
    # even when trajectories/ is missing (e.g. older exp_NNN dirs).
    #
    # Caveat: the before/after `.prom` files are captured via a single curl
    # against the vllm decode-worker Service, which load-balances across N
    # worker pods — so a single snapshot only contains one random pod's
    # counters. For multi-worker DGDs the diff approximates per-worker share,
    # not aggregate. We surface BOTH the raw single-pod numbers and an
    # implied-aggregate estimate (raw × N_workers, when N can be inferred
    # from per_request via instance count). Honest about the limitation.
    if prom_diff:
        ttft_sum_s = prom_diff.get("vllm:time_to_first_token_seconds_sum", 0.0)
        e2e_sum_s = prom_diff.get("vllm:e2e_request_latency_seconds_sum", 0.0)
        decode_sum_s = max(0.0, e2e_sum_s - ttft_sum_s)
        tool_sum_s = sum(
            r["tool_time_s"] for r in per_request if r["tool_time_s"] is not None
        )
        # Cross-check against per-request cumulative LLM time. If we have
        # nvext.timing data, that's the true aggregate; ratio = nvext_total /
        # prom_e2e ≈ replica count when the curl hit only one pod.
        cum_llm_ms = sum(r["llm_time_ms"] for r in per_request)
        implied_replicas: float | None = None
        if e2e_sum_s > 0 and cum_llm_ms > 0:
            implied_replicas = (cum_llm_ms / 1000.0) / e2e_sum_s

        lines += [
            "",
            "--- Time breakdown (vllm prom diff, server-side) ---",
            "# NOTE: prom snapshot is curled once against the worker Service —",
            "# for multi-worker DGDs (replicas > 1) this captures ONE random",
            "# pod's counters, not aggregate. See 'implied replicas' below.",
            f"Total e2e (Σ vllm:e2e_request_latency, 1 pod):  {e2e_sum_s:>10,.1f} s",
            f"  Total prefill (Σ vllm:time_to_first_token):   {ttft_sum_s:>10,.1f} s  "
            f"({100 * ttft_sum_s / e2e_sum_s if e2e_sum_s > 0 else 0:.2f}%)",
            f"  Total decode  (= e2e - prefill):              {decode_sum_s:>10,.1f} s  "
            f"({100 * decode_sum_s / e2e_sum_s if e2e_sum_s > 0 else 0:.2f}%)",
            f"Total tool execution (Σ tool_time_s):           {tool_sum_s:>10,.1f} s",
        ]
        if implied_replicas is not None:
            lines.append(
                f"Implied replicas (cum_llm / prom_e2e):          "
                f"{implied_replicas:>10.2f}   "
                f"(≈ DGD replica count; multiply prom totals by this for aggregate)"
            )
        if est_itl_ms is not None:
            lines.append(
                f"Global ITL (used for est_decode_ms):            {est_itl_ms:>10,.2f} ms/tok"
            )

    return "\n".join(lines) + "\n"


# =========================================================================
# Public entry point
# =========================================================================

def write_summary(exp_dir: Path, model_name: str | None = None) -> None:
    """Write `summary.txt` + `tool_call_timings.jsonl` for one experiment.

    Reads:
      - ``exp_dir/trajectory_collection.jsonl``
      - ``exp_dir/trajectories/results/verified/<model>/<instance>/*.traj.json``
      - ``exp_dir/trajectories/results/verified/<model>/<instance>/report_*.json``
      - ``exp_dir/qwen*-{before,after}-*.prom`` (optional; skip time
        breakdown when missing)

    When ``model_name`` is None, auto-detects from
    ``<exp>/manifests/recipe.yaml`` first, then any ``.traj.json``'s
    ``responses[0].model`` field (stripping the ``hosted_vllm/`` prefix).
    """
    exp_dir = Path(exp_dir).resolve()
    if model_name is None:
        model_name = _autodetect_model_from_manifests(exp_dir) or (
            _autodetect_model_from_trajectories(exp_dir / "trajectories")
        )
    if not model_name:
        print(f"[analysis] could not autodetect model; pass --model")
        return

    # `trajectory_collection.jsonl` is the NeMo-RL-side row-per-instance
    # file. It's present in exp_NNN dirs produced by run_dynamo_rollout_only,
    # but absent from standalone mini_swe_agent archives (e.g. the 5/19
    # production archive). Tolerate either case: when missing, fall back to
    # the trajectories/ tree alone, with rows=[] so the reward/repo block
    # collapses to "<no trajectory_collection.jsonl>".
    jsonl_path = exp_dir / "trajectory_collection.jsonl"
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        try:
            rows = [
                json.loads(l)
                for l in jsonl_path.read_text().splitlines()
                if l.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[analysis] failed to read {jsonl_path}: {exc}")
    else:
        print(f"[analysis] {jsonl_path} missing — reward/repo block will be empty")

    traj_root = exp_dir / "trajectories"
    per_request, steps_per_instance = _collect_per_request_stats(
        traj_root, model_name
    )
    outcomes = _collect_per_instance_outcomes(traj_root, model_name)
    prom_diff = _read_vllm_prom_diff(exp_dir)
    est_itl_ms = _compute_est_itl_ms(prom_diff)

    # ---- tool_call_timings.jsonl (with est_prefill_ms / est_decode_ms) ----
    if per_request:
        ttc_path = exp_dir / "tool_call_timings.jsonl"
        with ttc_path.open("w") as f:
            for rec in per_request:
                row = dict(rec)
                if est_itl_ms is not None:
                    est_decode = rec["completion_tokens"] * est_itl_ms
                    est_prefill = max(0.0, rec["llm_time_ms"] - est_decode)
                    row["est_prefill_ms"] = round(est_prefill, 2)
                    row["est_decode_ms"] = round(est_decode, 2)
                    row["est_itl_ms_used"] = round(est_itl_ms, 4)
                f.write(json.dumps(row) + "\n")
        print(f"[analysis] wrote {ttc_path} ({len(per_request)} rows)")

    # ---- summary.txt ----
    body = _render_summary(
        exp_dir=exp_dir,
        model_name=model_name,
        rows=rows,
        per_request=per_request,
        steps_per_instance=steps_per_instance,
        outcomes=outcomes,
        prom_diff=prom_diff,
        est_itl_ms=est_itl_ms,
    )
    (exp_dir / "summary.txt").write_text(body)
    print(f"[analysis] wrote {exp_dir / 'summary.txt'}")


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary.txt + tool_call_timings.jsonl for one exp_NNN.",
    )
    parser.add_argument(
        "exp_dir",
        help="path to an exp_NNN directory containing trajectory_collection.jsonl",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model name override (default: autodetect from manifests/recipe.yaml or any .traj.json)",
    )
    args = parser.parse_args()
    write_summary(Path(args.exp_dir), args.model)


if __name__ == "__main__":
    main()
