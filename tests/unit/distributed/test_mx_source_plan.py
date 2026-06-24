from nemo_rl.distributed.mx_source_plan import (
    make_source_plan,
    source_candidates_from_results,
)


def _candidate(source_id: str, version: int) -> dict:
    return {
        "format": "nemo_rl.mx_source_candidate.v1",
        "ref": {
            "mx_source_id": source_id,
            "worker_id": f"worker-{source_id}",
            "model_name": "model",
            "worker_rank": 0,
            "training_step": version,
        },
    }


def test_source_candidates_from_results_flattens_and_dedupes():
    candidate = _candidate("a", 2)
    result = [
        {"source_candidates": [candidate, candidate]},
        {"nested": "ignored"},
        [_candidate("b", 2)],
    ]

    assert source_candidates_from_results(result) == [candidate, _candidate("b", 2)]


def test_make_source_plan_keeps_exact_version_only():
    plan = make_source_plan(
        version=2,
        candidates=[_candidate("stale", 1), _candidate("exact", 2)],
        model_name="model",
    )

    assert plan["format"] == "nemo_rl.mx_source_plan.v1"
    assert plan["version"] == 2
    assert plan["model_name"] == "model"
    assert [candidate["ref"]["mx_source_id"] for candidate in plan["candidates"]] == [
        "exact"
    ]
