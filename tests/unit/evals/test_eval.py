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


import asyncio
import sys

from omegaconf import OmegaConf
import pytest
import torch

import examples.run_eval as run_eval
import nemo_rl.evals.eval as eval_mod
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.evals.eval import (
    _build_generation_inputs,
    _decode_generated_texts,
    _generate_outputs,
    _setup_generation,
    eval_cons_k,
    eval_pass_k,
)


class _FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row.tolist()) for row in token_ids]


def _base_generation_config(backend):
    return {
        "backend": backend,
        "model_name": "test-model",
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stop_token_ids": [0],
        "stop_strings": None,
        "num_prompts_per_step": -1,
        "_pad_token_id": 0,
    }


def test_run_eval_parse_args_uses_only_remaining_dotlist(monkeypatch):
    """CLI overrides should not re-parse --config as an OmegaConf key."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--config",
            "examples/configs/evals/eval_dynamo.yaml",
            "generation.backend=dynamo",
            "generation.dynamo_cfg.frontend_url=http://dynamo.example/v1",
        ],
    )

    args, overrides = run_eval.parse_args()

    assert args.config == "examples/configs/evals/eval_dynamo.yaml"
    assert OmegaConf.to_container(overrides, resolve=True) == {
        "generation": {
            "backend": "dynamo",
            "dynamo_cfg": {"frontend_url": "http://dynamo.example/v1"},
        }
    }


def test_setup_generation_uses_no_local_cluster_for_dynamo(monkeypatch):
    """Dynamo eval should only build the HTTP forwarder."""
    calls = []

    class FakeDynamoGeneration:
        def __init__(self, cluster, config):
            calls.append((cluster, config))
            self.dp_openai_server_base_urls = ["http://dynamo.example/v1"]

    def fail_cluster(*args, **kwargs):
        raise AssertionError("Dynamo eval must not allocate a RayVirtualCluster")

    monkeypatch.setattr(eval_mod, "DynamoGeneration", FakeDynamoGeneration)
    monkeypatch.setattr(eval_mod, "RayVirtualCluster", fail_cluster)

    config = _base_generation_config("dynamo")
    config["dynamo_cfg"] = {"frontend_url": "http://dynamo.example/v1"}

    generation = _setup_generation(
        generation_config=config,
        cluster_config={"gpus_per_node": 8, "num_nodes": 2},
    )

    assert generation.dp_openai_server_base_urls == ["http://dynamo.example/v1"]
    assert calls == [(None, config)]


def test_setup_generation_allocates_cluster_for_vllm(monkeypatch):
    """vLLM eval still owns local generation workers."""
    cluster_calls = []
    generation_calls = []

    class FakeCluster:
        def __init__(self, **kwargs):
            cluster_calls.append(kwargs)

    class FakeVllmGeneration:
        def __init__(self, cluster, config):
            generation_calls.append((cluster, config))

    monkeypatch.setattr(eval_mod, "RayVirtualCluster", FakeCluster)
    monkeypatch.setattr(eval_mod, "VllmGeneration", FakeVllmGeneration)

    config = _base_generation_config("vllm")
    config["vllm_cfg"] = {
        "async_engine": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
    }

    _setup_generation(
        generation_config=config,
        cluster_config={"gpus_per_node": 4, "num_nodes": 2},
    )

    assert cluster_calls == [
        {
            "name": "eval_cluster",
            "bundle_ct_per_node_list": [4, 4],
            "use_gpus": True,
            "num_gpus_per_node": 4,
            "max_colocated_worker_groups": 1,
        }
    ]
    assert len(generation_calls) == 1
    assert isinstance(generation_calls[0][0], FakeCluster)
    assert generation_calls[0][1] is config


def test_build_generation_inputs_flattens_message_log():
    batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "first",
                        "token_ids": torch.tensor([1, 2]),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "second",
                        "token_ids": torch.tensor([3]),
                    }
                ],
            ]
        }
    )

    inputs, input_lengths = _build_generation_inputs(
        batch=batch,
        tokenizer=_FakeTokenizer(),
        backend="dynamo",
    )

    assert inputs["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert input_lengths.tolist() == [2, 1]
    assert inputs["stop_strings"] == [None, None]


def test_build_generation_inputs_rejects_dynamo_multimodal():
    batch = BatchedDataDict(
        {
            "message_log": [
                [{"role": "user", "content": "look", "token_ids": torch.tensor([1])}]
            ],
            "vllm_content": ["<audio> prompt"],
        }
    )

    with pytest.raises(ValueError, match="text-only"):
        _build_generation_inputs(
            batch=batch,
            tokenizer=_FakeTokenizer(),
            backend="dynamo",
        )


def test_decode_generated_texts_uses_only_generated_suffix():
    outputs = BatchedDataDict(
        {
            "output_ids": torch.tensor([[1, 2, 7, 8, 0], [3, 9, 0, 0, 0]]),
            "generation_lengths": torch.tensor([2, 1]),
            "unpadded_sequence_lengths": torch.tensor([4, 2]),
            "logprobs": torch.zeros((2, 5)),
        }
    )

    texts = _decode_generated_texts(
        generation_outputs=outputs,
        input_lengths=torch.tensor([2, 1]),
        tokenizer=_FakeTokenizer(),
    )

    assert texts == ["7 8", "9"]


def test_generate_outputs_async_preserves_input_order():
    class FakeAsyncGeneration:
        async def generate_async(self, data, greedy=False):
            del greedy
            prompt_id = int(data["input_ids"][0, 0])
            if prompt_id == 1:
                await asyncio.sleep(0.01)
            output = BatchedDataDict(
                {
                    "output_ids": torch.tensor([[prompt_id, prompt_id + 10]]),
                    "generation_lengths": torch.tensor([1]),
                    "unpadded_sequence_lengths": torch.tensor([2]),
                    "logprobs": torch.zeros((1, 2)),
                }
            )
            yield 0, output

    inputs = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1], [2]]),
            "input_lengths": torch.tensor([1, 1]),
        }
    )

    outputs = asyncio.run(
        _generate_outputs(
            generation=FakeAsyncGeneration(),
            inputs=inputs,
            use_async=True,
            pad_token_id=0,
        )
    )

    assert outputs["output_ids"].tolist() == [[1, 11], [2, 12]]


def test_eval_pass_k_basic():
    """Test basic pass@k evaluation."""
    # Test case: 3 samples, 2 correct, k=1
    rewards = torch.tensor([1.0, 0.0, 1.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    group_size = len(rewards) / num_tests_per_prompt
    average_score = score / group_size
    expected = 2 / 3
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_all_correct():
    """Test pass@k when all samples are correct."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    group_size = len(rewards) / num_tests_per_prompt
    average_score = score / group_size
    expected = 1.0
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_none_correct():
    """Test pass@k when no samples are correct."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    average_score = score / (len(rewards) / num_tests_per_prompt)
    expected = 0.0
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_multiple_groups():
    """Test pass@k with multiple groups."""
    # Two groups: [1,0,1] and [0,1,0]
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    num_tests_per_prompt = 3
    score = eval_pass_k(rewards, num_tests_per_prompt=num_tests_per_prompt, k=1)
    average_score = score / (len(rewards) / num_tests_per_prompt)
    expected = 0.5
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_basic():
    """Test basic cons@k evaluation."""
    rewards = torch.tensor([1.0, 0.0, 1.0])
    extracted_answers = ["A", "B", "A"]
    num_tests_per_prompt = 3
    group_size = len(rewards) / num_tests_per_prompt
    score = eval_cons_k(
        rewards,
        num_tests_per_prompt=num_tests_per_prompt,
        k=1,
        extracted_answers=extracted_answers,
    )
    average_score = score / group_size
    expected = 2 / 3
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_multiple_groups():
    """Test cons@k with multiple groups."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    num_tests_per_prompt = 5
    extracted_answers = [
        "Correct",
        "Wrong1",
        "Correct",
        "Wrong2",
        "Correct",
        "Wrong3",
        "Correct",
        "Wrong4",
        "Correct",
        "Wrong4",
    ]
    group_size = len(rewards) / num_tests_per_prompt
    score = eval_cons_k(
        rewards,
        num_tests_per_prompt=num_tests_per_prompt,
        k=3,
        extracted_answers=extracted_answers,
    )
    average_score = score / group_size

    """
    For the first group, the extracted answers are [Correct, Wrong1, Correct, Wrong2, Correct]
    When calculating unbiased estimate of cons@3(k=3), we need to consider the majority vote of all Combination(5, 3) = 10 cases.
    The 10 cases are:
    - Correct, Wrong1, Correct      Majority: Correct
    - Correct, Wrong1, Wrong2       Majority: Correct(Choose the first one when there is a tie)
    - Correct, Wrong1, Correct      Majority: Correct
    - Correct, Correct, Wrong2      Majority: Correct
    - Correct, Correct, Correct     Majority: Correct
    - Correct, Wrong2, Correct      Majority: Correct
    - Wrong1, Correct, Wrong2       Majority: Wrong1 (Choose the first one when there is a tie)
    - Wrong1, Correct, Correct      Majority: Correct
    - Wrong1, Wrong2, Correct       Majority: Wrong1 (Choose the first one when there is a tie)
    - Correct, Wrong2, Correct      Majority: Correct
    The final result is 8/10.

    For the second group, the extracted answers are [Wrong3, Correct, Wrong4, Correct, Wrong4]
    When calculating unbiased estimate of cons@3(k=3), we need to consider the majority vote of all Combination(5, 3) = 10 cases.
    The 10 cases are:
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Correct, Correct      Majority: Correct
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Wrong4, Correct       Majority: Wrong3 (Choose the first one when there is a tie)
    - Wrong3, Wrong4, Wrong4        Majority: Wrong4
    - Wrong3, Correct, Wrong4       Majority: Wrong3 (Choose the first one when there is a tie)
    - Correct, Wrong4, Correct      Majority: Correct
    - Correct, Wrong4, Wrong4       Majority: Wrong4 (Choose the first one when there is a tie)
    - Correct, Correct, Wrong4      Majority: Correct
    - Wrong4, Correct, Wrong4       Majority: Wrong4
    The final result is 3/10.
    Since there len(rewards)/num_tests_per_prompt = 10/5 = 2 groups
    The final result is( 8/10 + 3/10 ) / 2 = 11/20 = 0.55
    """
    expected = 11 / 20
    assert isinstance(average_score, float)
    assert average_score == pytest.approx(expected, rel=1e-6)
