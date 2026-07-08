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

from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
SWE_CONFIG_DIR = REPO_ROOT / "examples" / "swe_bench"
DYNAMO_CONFIG = SWE_CONFIG_DIR / "grpo_nano_v3_5_swe_dynamo_hsg.yaml"
VLLM_CONFIG = SWE_CONFIG_DIR / "grpo_nano_v3_5_swe_vllm_hsg.yaml"
VLLM_ONLY_FIELDS = {
    "async_engine",
    "expose_http_server",
    "reasoning_parser_plugin",
    "http_server_serving_chat_kwargs",
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_swe_vllm_comparison_only_changes_backend_contract() -> None:
    dynamo = deepcopy(_load(DYNAMO_CONFIG))
    vllm = deepcopy(_load(VLLM_CONFIG))

    dynamo_generation = dynamo["policy"].pop("generation")
    vllm_generation = vllm["policy"].pop("generation")

    # Training, data, Gym, GRPO, logging, and cluster shape must remain exact.
    assert dynamo == vllm

    assert dynamo_generation.pop("backend") == "dynamo"
    assert vllm_generation.pop("backend") == "vllm"
    assert "dynamo_cfg" in dynamo_generation
    assert "dynamo_cfg" not in vllm_generation
    dynamo_generation.pop("dynamo_cfg")

    vllm_cfg = vllm_generation["vllm_cfg"]
    vllm_only = {field: vllm_cfg.pop(field) for field in VLLM_ONLY_FIELDS}

    # Sampling, TP/PP, memory, model length, vLLM kwargs, and placement match.
    assert dynamo_generation == vllm_generation

    assert vllm_only == {
        "async_engine": True,
        "expose_http_server": True,
        "reasoning_parser_plugin": (
            "nemo_rl/models/generation/vllm/reasoning_parsers/"
            "nano_v3_reasoning_parser.py"
        ),
        "http_server_serving_chat_kwargs": {
            "enable_auto_tools": True,
            "tool_parser": "qwen3_coder",
            "reasoning_parser": "nano_v3",
        },
    }
