# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import sys
import types
from types import SimpleNamespace

os.environ.setdefault("NRL_IGNORE_VERSION_MISMATCH", "1")
sys.modules.setdefault("vllm", types.ModuleType("vllm"))

from nemo_rl.models.generation.vllm import vllm_backend


def _candidate(
    *,
    role: str,
    version: int,
    worker_rank: int,
    source_id: str,
    tp_rank: int | None = None,
    tp_size: int | None = None,
) -> SimpleNamespace:
    megatron_meta = None
    if tp_rank is not None and tp_size is not None:
        megatron_meta = SimpleNamespace(tp_rank=tp_rank, tp_size=tp_size)
    return SimpleNamespace(
        role=role,
        worker_rank=worker_rank,
        megatron_meta=megatron_meta,
        ref=SimpleNamespace(
            training_step=version,
            mx_source_id=source_id,
            worker_id=f"worker-{source_id}",
        ),
    )


def test_choose_megatron_bulk_source_requires_exact_version():
    candidates = [
        _candidate(
            role="trainer",
            version=4,
            worker_rank=0,
            source_id="stale",
            tp_rank=0,
            tp_size=1,
        ),
        _candidate(
            role="trainer",
            version=6,
            worker_rank=0,
            source_id="future",
            tp_rank=0,
            tp_size=1,
        ),
        _candidate(
            role="trainer",
            version=5,
            worker_rank=0,
            source_id="exact",
            tp_rank=0,
            tp_size=1,
        ),
    ]

    chosen, stats = vllm_backend._choose_megatron_bulk_source(
        candidates,
        version=5,
        target_tp_rank=0,
        target_tp_size=1,
        selector_key="target-0",
    )

    assert chosen.ref.mx_source_id == "exact"
    assert stats["stale_version"] == 1
    assert stats["future_version"] == 1
    assert stats["eligible_trainers"] == 1
    assert stats["selection_pool"] == "trainer"
    assert stats["selection_pool_size"] == 1


def test_choose_megatron_bulk_source_prefers_replicas_over_trainers():
    candidates = [
        _candidate(
            role="trainer",
            version=5,
            worker_rank=0,
            source_id="trainer-0",
            tp_rank=0,
            tp_size=1,
        ),
        _candidate(
            role="trainer",
            version=5,
            worker_rank=0,
            source_id="trainer-1",
            tp_rank=0,
            tp_size=1,
        ),
        _candidate(
            role="inference_replica",
            version=5,
            worker_rank=0,
            source_id="replica-0",
        ),
        _candidate(
            role="inference_replica",
            version=5,
            worker_rank=0,
            source_id="replica-1",
        ),
    ]

    chosen, stats = vllm_backend._choose_megatron_bulk_source(
        candidates,
        version=5,
        target_tp_rank=0,
        target_tp_size=1,
        selector_key="target-0",
    )

    assert chosen.ref.mx_source_id in {"replica-0", "replica-1"}
    assert stats["eligible_trainers"] == 2
    assert stats["eligible_replicas"] == 2
    assert stats["selection_pool"] == "inference_replica"
    assert stats["selection_pool_size"] == 2


def test_choose_megatron_bulk_source_spreads_trainer_fallback_by_target():
    candidates = [
        _candidate(
            role="trainer",
            version=5,
            worker_rank=0,
            source_id=f"trainer-{idx}",
            tp_rank=0,
            tp_size=1,
        )
        for idx in range(4)
    ]

    chosen_ids = {
        vllm_backend._choose_megatron_bulk_source(
            candidates,
            version=5,
            target_tp_rank=0,
            target_tp_size=1,
            selector_key=f"target-{idx}",
        )[0].ref.mx_source_id
        for idx in range(32)
    }

    assert chosen_ids == {"trainer-0", "trainer-1", "trainer-2", "trainer-3"}


def test_choose_megatron_bulk_source_filters_incompatible_tp_layout():
    candidates = [
        _candidate(
            role="trainer",
            version=5,
            worker_rank=1,
            source_id="wrong-trainer-rank",
            tp_rank=1,
            tp_size=1,
        ),
        _candidate(
            role="inference_replica",
            version=5,
            worker_rank=1,
            source_id="wrong-replica-rank",
        ),
        _candidate(
            role="trainer",
            version=5,
            worker_rank=0,
            source_id="trainer-0",
            tp_rank=0,
            tp_size=1,
        ),
    ]

    chosen, stats = vllm_backend._choose_megatron_bulk_source(
        candidates,
        version=5,
        target_tp_rank=0,
        target_tp_size=1,
        selector_key="target-0",
    )

    assert chosen.ref.mx_source_id == "trainer-0"
    assert stats["eligible_trainers"] == 1
    assert stats["eligible_replicas"] == 0
    assert stats["selection_pool"] == "trainer"
    assert stats["selection_pool_size"] == 1


def test_choose_megatron_bulk_source_returns_none_without_exact_eligible_source():
    candidates = [
        _candidate(
            role="trainer",
            version=5,
            worker_rank=0,
            source_id="wrong-tp-size",
            tp_rank=0,
            tp_size=2,
        ),
        _candidate(
            role="inference_replica",
            version=4,
            worker_rank=0,
            source_id="stale-replica",
        ),
    ]

    chosen, stats = vllm_backend._choose_megatron_bulk_source(
        candidates,
        version=5,
        target_tp_rank=0,
        target_tp_size=1,
        selector_key="target-0",
    )

    assert chosen is None
    assert stats["same_version"] == 1
    assert stats["eligible_trainers"] == 0
    assert stats["eligible_replicas"] == 0
    assert stats["selection_pool"] == "none"
    assert stats["selection_pool_size"] == 0
