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
"""Nemotron4 (NM4) support for the Megatron backend.

Importing this package registers both halves of what an NM4 refit needs: the
HF config class, so ``AutoConfig`` can read an NM4 export, and the weight
bridge, so ``AutoBridge.export_hf_weights`` can name NM4 tensors.

The weight mappings come from Megatron-Bridge's own ``Nemotron4Bridge``. NeMo RL
carried a duplicate registry until a 32-node refit showed the two agree
numerically (``token_mult_prob_error`` within 0.001 at step 1), so only the two
pieces Megatron-Bridge cannot supply on its own are left here.
"""

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.experimental_nm4_llava_provider import NM4LlavaModel
from megatron.bridge.models.nemotron4 import Nemotron4Bridge as _Nemotron4Bridge

from nemo_rl.models.huggingface.nemotron4 import register_nemotron4_config

register_nemotron4_config()

# Megatron-Bridge registers this bridge against Nemotron4VLModel, which its own
# Nemotron4VLModelProvider builds from an HF config. We instead build the model
# with ExperimentalNM4LlavaProvider from the checkpoint's saved args, giving an
# NM4LlavaModel, and stream_weights_megatron_to_hf dispatches on the
# (architecture, model class) pair -- so without this second registration a
# refit raises NotImplementedError. Registration is last-writer-wins, and the
# two targets are distinct keys, so this adds to their registration rather than
# replacing it.
Nemotron4Bridge = MegatronModelBridge.register_bridge(
    source="Nemotron4ForConditionalGeneration",
    target=NM4LlavaModel,
    model_type="nemotron4",
)(_Nemotron4Bridge)

__all__ = ["Nemotron4Bridge", "register_nemotron4_config"]
