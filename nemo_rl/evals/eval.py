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
import json
import os
from collections import Counter
from itertools import combinations
from typing import Any, NotRequired, TypedDict, cast

import ray
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import EvalDataConfigType
from nemo_rl.data.collate_fn import eval_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.environments.vlm_environment import VLMEnvConfig
from nemo_rl.models.generation.dynamo import DynamoCfg, DynamoConfig, DynamoGeneration
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.generation.vllm.config import VllmSpecificArgs
from nemo_rl.models.policy import TokenizerConfig

# ===============================================================================
# Configuration
# ===============================================================================


class EvalConfig(TypedDict):
    metric: str
    num_tests_per_prompt: int
    seed: int
    k_value: int
    save_path: str | None


class EvalGenerationConfig(GenerationConfig):
    """Generation config fields consumed by the eval entrypoint."""

    num_prompts_per_step: int
    vllm_cfg: NotRequired[VllmSpecificArgs]
    vllm_kwargs: NotRequired[dict[str, Any]]
    dynamo_cfg: NotRequired[DynamoCfg]


# TODO: this should updated, but is left to avoid breaking changes
class _PassThroughEnvConfig(TypedDict):
    math: NotRequired[MathEnvConfig]
    mmau: NotRequired[VLMEnvConfig]


class MasterConfig(BaseModel, extra="allow"):
    eval: EvalConfig
    generation: EvalGenerationConfig  # Fixed: was 'generate'
    tokenizer: TokenizerConfig  # Added missing tokenizer key
    data: EvalDataConfigType
    env: _PassThroughEnvConfig
    cluster: ClusterConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> tuple[
    GenerationInterface,
    DataLoader,
    MasterConfig,
]:
    """Set up components for model evaluation.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, and config.
    """
    # Extract individual configs for easier access
    eval_config = master_config.eval
    generation_config = master_config.generation
    cluster_config = master_config.cluster

    # Set seed for reproducibility
    set_seed(eval_config["seed"])

    # Check settings
    metric = eval_config["metric"]
    k_value = eval_config["k_value"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    temperature = generation_config["temperature"]
    top_k = generation_config["top_k"]

    # Validate metrics
    assert metric in ["pass@k", "cons@k"], f"Invalid metric: {metric}"
    if num_tests_per_prompt > 1:
        assert temperature > 0 and top_k != 1, (
            "temperature > 0 and top_k != 1 are required for multiple samples"
        )

    assert k_value >= 1, "k_value must be greater than or equal to 1"
    assert num_tests_per_prompt >= k_value, (
        "num_tests_per_prompt must be greater than or equal to k_value "
    )

    # ==========================
    #           Data
    # ==========================
    if generation_config["num_prompts_per_step"] == -1:
        generation_config["num_prompts_per_step"] = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=generation_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    print(f"  ✓ Evaluation dataset loaded with {len(dataset)} samples")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    generation = _setup_generation(
        generation_config=generation_config,
        cluster_config=cluster_config,
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        generation,
        dataloader,
        master_config,
    )


def _setup_generation(
    generation_config: EvalGenerationConfig,
    cluster_config: ClusterConfig,
) -> GenerationInterface:
    """Set up the configured eval generation backend."""
    backend = generation_config["backend"]
    if backend == "vllm":
        print("\n▶ Setting up compute cluster...")
        cluster = RayVirtualCluster(
            name="eval_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1,
        )
        print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")
        generation = VllmGeneration(
            cluster=cluster,
            config=cast(VllmConfig, generation_config),
        )
        print(
            f"  ✓ Using vLLM backend for generation with "
            f"{generation_config['model_name']}"
        )
        return generation

    if backend == "dynamo":
        generation = DynamoGeneration(
            cluster=None,
            config=cast(DynamoConfig, generation_config),
        )
        frontend_url = generation.dp_openai_server_base_urls[0]
        print(f"  ✓ Using Dynamo backend (frontend: {frontend_url})")
        return generation

    raise ValueError(
        f"Unsupported evaluation generation backend: {backend}. "
        "Supported backends are: vllm, dynamo."
    )


# ===============================================================================
# Evaluation
# ===============================================================================


def eval_pass_k(rewards: torch.Tensor, num_tests_per_prompt: int, k: int) -> float:
    """Evaluate pass@k score using an unbiased estimator.

    Reference: https://github.com/huggingface/evaluate/blob/32546aafec25cdc2a5d7dd9f941fc5be56ba122f/metrics/code_eval/code_eval.py#L198-L213
    Args:
        rewards: Tensor of shape (batch_size * num_tests_per_prompt)
        k: int (pass@k value)

    Returns:
        pass_k_score: float
    """

    def eval_single_chunk(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return float(1.0 - torch.prod(1.0 - k / torch.arange(n - c + 1, n + 1)).item())

    # rewards is a 1d tensor of size (batch_size * num_tests_per_prompt)
    group_rewards = rewards.split(num_tests_per_prompt)
    pass_k_score = 0.0
    for group_reward in group_rewards:
        num_correct = group_reward.sum().item()
        pass_k_score += eval_single_chunk(num_tests_per_prompt, num_correct, k)

    return pass_k_score


def eval_cons_k(
    rewards: torch.Tensor,
    num_tests_per_prompt: int,
    k: int,
    extracted_answers: list[str | None],
) -> float:
    """Evaluate cons@k score using an unbiased estimator.

    Args:
        rewards: Tensor of shape (batch_size * num_tests_per_prompt)
        num_tests_per_prompt: int
        k: int
        extracted_answers: list[str| None]

    Returns:
        cons_k_score: float
    """

    def majority_vote(answers: list[str | None]) -> str | None:
        """Find the most common answer in the list of answers."""
        if not answers:
            return None
        # To fix@rayentian: How to deal with the case that there are multiple most common answers? Now we just return the first one.
        return Counter(answers).most_common(1)[0][0]

    def eval_single_cons_k(
        chunk_rewards: torch.Tensor, chunk_answers: list[str | None], n: int, k: int
    ) -> float:
        if chunk_answers is None or n == 0 or k > n:
            return 0.0

        total_subsets = 0
        correct_subsets = 0
        # For each subset of k answers, we vote for the most common answer.
        # If the most common answer is the same as the gold answer, we consider the subset as correct.
        for subset_indices in combinations(range(n), k):
            subset_answers = [chunk_answers[i] for i in subset_indices]
            majority_answer = majority_vote(subset_answers)
            reward_idx = chunk_answers.index(majority_answer)
            reward = chunk_rewards[reward_idx].item()
            total_subsets += 1
            if reward == 1.0:
                correct_subsets += 1

        return correct_subsets / total_subsets

    assert len(extracted_answers) == len(rewards), (
        "The number of extracted answers must be the same as the number of rewards"
    )
    # Split the rewards and extracted answers into groups of num_tests_per_prompt.
    group_rewards = rewards.split(num_tests_per_prompt)
    group_extracted_answers = [
        extracted_answers[i : i + num_tests_per_prompt]
        for i in range(0, len(extracted_answers), num_tests_per_prompt)
    ]
    assert len(group_rewards) == len(group_extracted_answers), (
        "The number of rewards and extracted answers must be the same"
    )
    num_groups = len(group_rewards)
    cons_k_score = 0.0
    # For each group of num_tests_per_prompt rewards and extracted answers, we evaluate the cons@k score.
    for i in range(num_groups):
        chunk_rewards = group_rewards[i]
        chunk_answers = group_extracted_answers[i]
        assert len(chunk_rewards) == len(chunk_answers), (
            "The number of rewards and extracted answers must be the same"
        )
        cons_k_score += eval_single_cons_k(
            chunk_rewards, chunk_answers, len(chunk_answers), k
        )

    return cons_k_score


def run_env_eval(generation, dataloader, env, master_config, tokenizer):
    """Main entry point for running evaluation using environment.

    Generates model responses and evaluates them by env.

    Args:
        generation: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        env: Environment that scores responses.
        master_config: Configuration settings.
        tokenizer: Tokenizer used to decode generated token IDs.
    """
    use_async = _should_use_async_generation(generation, master_config.generation)
    asyncio.run(
        _run_env_eval_impl(
            generation,
            dataloader,
            env,
            master_config,
            tokenizer,
            use_async=use_async,
        )
    )


async def _run_env_eval_impl(
    generation,
    dataloader,
    env,
    master_config,
    tokenizer,
    use_async=False,
):
    """Unified implementation for both sync and async evaluation."""
    # Extract for easier access
    generation_config = master_config.generation
    eval_config = master_config.eval
    metric = eval_config["metric"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    k_value = eval_config["k_value"]

    # List to collect evaluation data for parquet file
    evaluation_data = []

    # Run evaluation loop
    score = 0.0
    for batch in dataloader:
        # measure multiple samples
        if num_tests_per_prompt > 1:
            batch = batch.repeat_interleave(num_tests_per_prompt)

        # get input prompt from message_log
        is_multimodal = "vllm_content" in batch
        prompts = []
        prompts_for_display = []
        for i, message_log in enumerate(batch["message_log"]):
            if is_multimodal and batch["vllm_content"][i] is not None:
                vllm_content = batch["vllm_content"][i]
                prompt_dict = {"prompt": vllm_content}
                multi_modal_data = {}
                audios = batch.get("vllm_audios", None)
                if audios is not None and len(audios[i]) > 0:
                    multi_modal_data["audio"] = (
                        audios[i][0] if len(audios[i]) == 1 else audios[i]
                    )
                images = batch.get("vllm_images", None)
                if images is not None and len(images[i]) > 0:
                    multi_modal_data["image"] = (
                        images[i][0] if len(images[i]) == 1 else images[i]
                    )
                if multi_modal_data:
                    prompt_dict["multi_modal_data"] = multi_modal_data
                prompts.append(prompt_dict)
                prompts_for_display.append(vllm_content)
            else:
                # Text-only fallback: use raw prompt strings (vLLM will tokenize them).
                # Note: utils.py's format_prompt_for_vllm_generation uses pre-tokenized
                # prompt_token_ids instead, since the training pipeline already has
                # input_ids tensors. Both are valid vLLM inputs but may tokenize
                # slightly differently.
                content = [message["content"] for message in message_log]
                content = "\n".join(content)
                prompts.append(content)
                prompts_for_display.append(content)

        generation_inputs, input_lengths = _build_generation_inputs(
            batch=batch,
            tokenizer=tokenizer,
            backend=generation_config["backend"],
        )
        generation_outputs = await _generate_outputs(
            generation=generation,
            inputs=generation_inputs,
            use_async=use_async,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs = _decode_generated_texts(
            generation_outputs=generation_outputs,
            input_lengths=input_lengths,
            tokenizer=tokenizer,
        )

        # append to message_log
        for idx, output in enumerate(outputs):
            batch["message_log"][idx].append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

        # evaluate generations with the environment
        to_env = [
            get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
            for i in range(len(batch["message_log"]))
        ]

        env_return = ray.get(env.step.remote(to_env, batch["extra_env_info"], True))
        rewards = env_return.rewards

        # Collect data for JSON file
        for i, (prompt, output, message_log, reward, extra_info) in enumerate(
            zip(
                prompts_for_display,
                outputs,
                batch["message_log"],
                rewards.tolist(),
                batch["extra_env_info"],
            )
        ):
            evaluation_data.append(
                {
                    "prompt": prompt,
                    "response": output,
                    "reward": reward,
                    "message_log": message_log,
                    "extra_env_info": extra_info,
                    "sample_index": len(evaluation_data),
                }
            )

        # update stats
        if metric == "pass@k":
            score += eval_pass_k(rewards, num_tests_per_prompt, k_value)
        elif metric == "cons@k":
            extracted_answers = env_return.answers
            score += eval_cons_k(
                rewards, num_tests_per_prompt, k_value, extracted_answers
            )
        else:
            raise ValueError(f"Invalid metric: {metric}")

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    generation.shutdown()

    # Save evaluation data to JSON file if save_path is specified
    save_path = eval_config.get("save_path")
    if evaluation_data and save_path is not None:
        _save_evaluation_data_to_json(evaluation_data, master_config, save_path)

    # Print results
    _print_results(
        master_config,
        generation_config,
        score,
        len(dataloader.dataset),
        metric,
        k_value,
        num_tests_per_prompt,
    )


def _should_use_async_generation(
    generation: GenerationInterface,
    generation_config: EvalGenerationConfig,
) -> bool:
    """Return whether eval should fan out per-sample async generation."""
    backend = generation_config["backend"]
    if backend == "dynamo":
        return True
    if backend == "vllm":
        vllm_cfg = generation_config["vllm_cfg"]
        return bool(vllm_cfg["async_engine"])
    return hasattr(generation, "generate_async")


def _build_generation_inputs(
    *,
    batch: BatchedDataDict[Any],
    tokenizer: AutoTokenizer,
    backend: str,
) -> tuple[BatchedDataDict[GenerationDatumSpec], torch.Tensor]:
    """Build GenerationInterface inputs from an eval batch."""
    if backend == "dynamo" and _has_multimodal_content(batch):
        raise ValueError(
            "Dynamo evaluation supports text-only datasets. Use backend=vllm "
            "for multimodal eval datasets such as MMAU."
        )

    flat_messages, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
    )
    inputs = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
        }
    )

    if "stop_strings" in batch:
        inputs["stop_strings"] = batch["stop_strings"]
    else:
        inputs["stop_strings"] = [None] * len(input_lengths)

    if backend == "vllm" and "vllm_content" in batch:
        inputs["vllm_content"] = batch["vllm_content"]
        if "vllm_images" in batch:
            inputs["vllm_images"] = batch["vllm_images"]
        if "vllm_audios" in batch:
            inputs["vllm_audios"] = batch["vllm_audios"]

    return inputs, input_lengths


def _has_multimodal_content(batch: BatchedDataDict[Any]) -> bool:
    """Return True if the batch contains vLLM multimodal prompt content."""
    if "vllm_content" not in batch:
        return False
    return any(content is not None for content in batch["vllm_content"])


async def _generate_outputs(
    *,
    generation: GenerationInterface,
    inputs: BatchedDataDict[GenerationDatumSpec],
    use_async: bool,
    pad_token_id: int,
) -> BatchedDataDict[GenerationOutputSpec]:
    """Generate token outputs using either sync or async backend methods."""
    if use_async:
        async def _generate_single_sample(i):
            single = inputs.slice(i, i + 1)
            async for _, result in generation.generate_async(single):
                return (i, result)
            raise RuntimeError(f"No output produced for sample {i}")

        results = await asyncio.gather(
            *(_generate_single_sample(i) for i in range(inputs.size))
        )
        results.sort(key=lambda x: x[0])
        return BatchedDataDict.from_batches(
            [result for _, result in results],
            pad_value_dict={"output_ids": pad_token_id, "logprobs": 0.0},
        )

    return generation.generate(inputs, greedy=False)


def _decode_generated_texts(
    *,
    generation_outputs: BatchedDataDict[GenerationOutputSpec],
    input_lengths: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> list[str]:
    """Decode only generated suffix tokens from generation outputs."""
    output_ids = generation_outputs["output_ids"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]

    generated_ids = []
    for i in range(len(input_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        generated_ids.append(output_ids[i, input_len:total_length])

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def _save_evaluation_data_to_json(evaluation_data, master_config, save_path):
    """Save evaluation data to a JSON file.

    Args:
        evaluation_data: List of evaluation samples
        master_config: Configuration dictionary
        save_path: Path to save evaluation results. Set to null to disable saving.
                  Example: "results/eval_output" or "/path/to/evaluation_results"
    """
    # Extract configuration information
    config_data = {
        "model_name": master_config.generation["model_name"],
        "dataset_name": master_config.data["dataset_name"],
        "metric": master_config.eval["metric"],
        "k_value": master_config.eval["k_value"],
        "num_tests_per_prompt": master_config.eval["num_tests_per_prompt"],
        "temperature": master_config.generation["temperature"],
        "top_p": master_config.generation["top_p"],
        "top_k": master_config.generation["top_k"],
    }

    # Create directory if it doesn't exist
    save_dir = save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Generate file paths within the directory
    eval_data_path = os.path.join(save_dir, "evaluation_data.json")
    config_path = os.path.join(save_dir, "config.json")

    # Prepare the data to save
    data_to_save = {"evaluation_data": evaluation_data}

    # Save configuration to separate JSON file
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")

    # Process data to make it JSON serializable
    processed_data = []
    for sample in evaluation_data:
        processed_sample = sample.copy()
        # Convert non-serializable objects to strings
        processed_sample["message_log"] = str(sample["message_log"])
        processed_sample["extra_env_info"] = str(sample["extra_env_info"])
        processed_data.append(processed_sample)

    # Update data to save with processed version
    data_to_save["evaluation_data"] = processed_data

    # Save to JSON file
    with open(eval_data_path, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\n✓ Evaluation data saved to: {eval_data_path}")
    print(f"  Total samples: {len(evaluation_data)}")
    print(f"  File size: {os.path.getsize(eval_data_path) / 1024 / 1024:.2f} MB")


def _print_results(
    master_config,
    generation_config,
    score,
    dataset_size,
    metric,
    k_value,
    num_tests_per_prompt,
):
    """Print evaluation results."""
    dataset_name = os.path.basename(master_config.data["dataset_name"])
    model_name = os.path.basename(generation_config["model_name"])
    max_new_tokens = generation_config["max_new_tokens"]
    seed = master_config.eval["seed"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]
    average_score = score / dataset_size

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"{max_new_tokens=} {temperature=} {top_p=} {top_k=} {seed=}\n")
    print(f"metric={metric[:-1]}{k_value} {num_tests_per_prompt=}\n")
    print(f"score={average_score:.4f} ({score}/{dataset_size})")
    print("=" * 60 + "\n")
