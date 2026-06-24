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

import math
import os
import time
from functools import lru_cache
from typing import Any, List, Tuple

import torch


@lru_cache(maxsize=1)
def get_target_packed_tensor_size():
    memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.02")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_memory_bytes = props.total_memory
    # max size is 5GB
    target_size = min(int(total_memory_bytes * float(memory_ratio)), 5 * 1024**3)
    return target_size


@lru_cache(maxsize=1)
def get_num_buffers():
    return int(os.getenv("NRL_REFIT_NUM_BUFFERS", "2"))


def packed_broadcast_producer(iterator, group, src, post_iter_func):
    """Broadcast a list of tensors in a packed manner.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_iter_func: function to apply to each tensor before packing, should return a tensor

    Returns:
        None

    """
    target_packed_tensor_size = get_target_packed_tensor_size()

    num_buffers = get_num_buffers()
    total_start = time.perf_counter()
    total_bytes = 0
    total_chunks = 0
    print(
        "[weight-sync-debug][packed-nccl-producer] "
        f"target_chunk_bytes={target_packed_tensor_size} num_buffers={num_buffers}",
        flush=True,
    )
    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_list = [[] for _ in range(num_buffers)]
    packing_tensor_sizes = [0 for _ in range(num_buffers)]
    packed_tensors = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        # Start tasks for the new buffer in a new stream
        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            try:
                # Initialize the packing tensor list and sizes
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                # Pack the tensors
                while True:
                    # Apply backend specific post processing and then convert to linearized uint8 tensor
                    tensor = post_iter_func(next(iterator)).view(torch.uint8).view(-1)
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.view(torch.uint8).numel()
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                # Pack the tensors and call broadcast collective
                cat_start = time.perf_counter()
                packed_tensors[buffer_idx] = torch.cat(
                    packing_tensor_list[buffer_idx], dim=0
                )
                cat_s = time.perf_counter() - cat_start
                broadcast_start = time.perf_counter()
                group.broadcast(packed_tensors[buffer_idx], src=src)
                broadcast_s = time.perf_counter() - broadcast_start
                chunk_bytes = packed_tensors[buffer_idx].numel()
                total_bytes += chunk_bytes
                total_chunks += 1
                print(
                    "[weight-sync-debug][packed-nccl-producer] "
                    f"chunk={total_chunks} bytes={chunk_bytes} "
                    f"cat_s={cat_s:.3f} broadcast_s={broadcast_s:.3f}",
                    flush=True,
                )
            except StopIteration:
                # do the last broadcast if there are remaining tensors
                if len(packing_tensor_list[buffer_idx]) > 0:
                    cat_start = time.perf_counter()
                    packed_tensors[buffer_idx] = torch.cat(
                        packing_tensor_list[buffer_idx], dim=0
                    )
                    cat_s = time.perf_counter() - cat_start
                    broadcast_start = time.perf_counter()
                    group.broadcast(packed_tensors[buffer_idx], src=src)
                    broadcast_s = time.perf_counter() - broadcast_start
                    chunk_bytes = packed_tensors[buffer_idx].numel()
                    total_bytes += chunk_bytes
                    total_chunks += 1
                    print(
                        "[weight-sync-debug][packed-nccl-producer] "
                        f"chunk={total_chunks} bytes={chunk_bytes} "
                        f"cat_s={cat_s:.3f} broadcast_s={broadcast_s:.3f}",
                        flush=True,
                    )
                break
    total_s = time.perf_counter() - total_start
    print(
        "[weight-sync-debug][packed-nccl-producer] "
        f"chunks={total_chunks} total_gb={total_bytes / 1e9:.3f} "
        f"total_s={total_s:.3f} gbps="
        f"{(total_bytes * 8) / (total_s * 1e9) if total_s > 0 else 0.0:.1f}",
        flush=True,
    )


def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
    """Consume a packed tensor and unpack it into a list of tensors.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_unpack_func: function to apply to each tensor after unpacking

    Returns:
        None

    """

    def unpack_tensor(
        packed_tensor: torch.Tensor, meta_data_list: list[Any]
    ) -> List[Tuple[str, torch.Tensor]]:
        """Unpack a single tensor into a list of tensors.

        Args:
            packed_tensor: the packed torch.uint8 tensor to unpack
            meta_data_list: List[(name, shape, dtype, offset, tensor_size)]

        Returns:
            unpacked List[(name, tensor)]
        """
        unpacked_list = []
        # Perform batched split with torch.split_with_sizes
        packed_tensor_sizes = list(map(lambda x: x[4], meta_data_list))
        unpacked_tensor = packed_tensor.split_with_sizes(packed_tensor_sizes)

        # unpacked_list = List[(name, torch.Tensor.view(dtype).view(*shape))]
        unpacked_list = [
            (
                meta_data_list[i][0],
                tensor.view(meta_data_list[i][2]).view(*meta_data_list[i][1]),
            )
            for i, tensor in enumerate(unpacked_tensor)
        ]

        return unpacked_list

    target_packed_tensor_size = get_target_packed_tensor_size()

    num_buffers = get_num_buffers()
    total_start = time.perf_counter()
    total_bytes = 0
    total_chunks = 0
    print(
        "[weight-sync-debug][packed-nccl-consumer] "
        f"target_chunk_bytes={target_packed_tensor_size} num_buffers={num_buffers}",
        flush=True,
    )
    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_meta_data = [[] for _ in range(num_buffers)]
    packing_tensor_sizes = [0 for _ in range(num_buffers)]
    offsets = [0 for _ in range(num_buffers)]
    packed_tensors = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            # Initialize the packing tensor meta data
            packing_tensor_meta_data[buffer_idx] = []
            packing_tensor_sizes[buffer_idx] = 0
            offsets[buffer_idx] = 0
            try:
                # Form a packed tensor
                while True:
                    name, (shape, dtype) = next(iterator)
                    tensor_size = math.prod(shape) * dtype.itemsize
                    packing_tensor_meta_data[buffer_idx].append(
                        (name, shape, dtype, offsets[buffer_idx], tensor_size)
                    )
                    packing_tensor_sizes[buffer_idx] += tensor_size
                    offsets[buffer_idx] += tensor_size
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                # Create a packed tensor and broadcast it
                alloc_start = time.perf_counter()
                packed_tensors[buffer_idx] = torch.empty(
                    packing_tensor_sizes[buffer_idx], dtype=torch.uint8, device="cuda"
                )
                alloc_s = time.perf_counter() - alloc_start
                broadcast_start = time.perf_counter()
                group.broadcast(packed_tensors[buffer_idx], src=src)
                broadcast_s = time.perf_counter() - broadcast_start
                # Load the packed tensor into the model
                unpack_start = time.perf_counter()
                unpacked = unpack_tensor(
                    packed_tensors[buffer_idx], packing_tensor_meta_data[buffer_idx]
                )
                unpack_s = time.perf_counter() - unpack_start
                load_start = time.perf_counter()
                post_unpack_func(unpacked)
                load_s = time.perf_counter() - load_start
                chunk_bytes = packing_tensor_sizes[buffer_idx]
                total_bytes += chunk_bytes
                total_chunks += 1
                print(
                    "[weight-sync-debug][packed-nccl-consumer] "
                    f"chunk={total_chunks} bytes={chunk_bytes} "
                    f"alloc_s={alloc_s:.3f} broadcast_s={broadcast_s:.3f} "
                    f"unpack_s={unpack_s:.3f} load_s={load_s:.3f}",
                    flush=True,
                )
            except StopIteration:
                # do the last broadcast if there are remaining tensors
                if len(packing_tensor_meta_data[buffer_idx]) > 0:
                    # Create a packed tensor and broadcast it
                    alloc_start = time.perf_counter()
                    packed_tensors[buffer_idx] = torch.empty(
                        packing_tensor_sizes[buffer_idx],
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    alloc_s = time.perf_counter() - alloc_start
                    broadcast_start = time.perf_counter()
                    group.broadcast(packed_tensors[buffer_idx], src=src)
                    broadcast_s = time.perf_counter() - broadcast_start
                    # Load the packed tensor into the model
                    unpack_start = time.perf_counter()
                    unpacked = unpack_tensor(
                        packed_tensors[buffer_idx],
                        packing_tensor_meta_data[buffer_idx],
                    )
                    unpack_s = time.perf_counter() - unpack_start
                    load_start = time.perf_counter()
                    post_unpack_func(unpacked)
                    load_s = time.perf_counter() - load_start
                    chunk_bytes = packing_tensor_sizes[buffer_idx]
                    total_bytes += chunk_bytes
                    total_chunks += 1
                    print(
                        "[weight-sync-debug][packed-nccl-consumer] "
                        f"chunk={total_chunks} bytes={chunk_bytes} "
                        f"alloc_s={alloc_s:.3f} broadcast_s={broadcast_s:.3f} "
                        f"unpack_s={unpack_s:.3f} load_s={load_s:.3f}",
                        flush=True,
                    )
                break
    total_s = time.perf_counter() - total_start
    print(
        "[weight-sync-debug][packed-nccl-consumer] "
        f"chunks={total_chunks} total_gb={total_bytes / 1e9:.3f} "
        f"total_s={total_s:.3f} gbps="
        f"{(total_bytes * 8) / (total_s * 1e9) if total_s > 0 else 0.0:.1f}",
        flush=True,
    )
