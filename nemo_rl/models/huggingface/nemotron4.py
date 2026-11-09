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
"""HuggingFace configuration for Nemotron4 (NM4).

The NM4 HF exports carry ``model_type: nemotron4`` but ship no modeling code
and no ``auto_map``, so ``AutoConfig.from_pretrained`` cannot read them on its
own. The NM4 vLLM fork registers an equivalent class, but only once vLLM is
imported and only in venvs that have it, which is too late and too narrow:
the Megatron workers need it to build an ``AutoBridge`` against an NM4 export,
and the vLLM workers need it before vLLM loads, because
``ModelFlag.VLLM_LOAD_FORMAT_AUTO`` reads the config first. Hence this copy,
which lives beside the other backend-agnostic HF helpers.

Kept deliberately field-for-field compatible with the vLLM fork's
``vllm/transformers_utils/configs/nemotron4.py`` so the two agree on what a
checkpoint means.
"""

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.pixtral import PixtralVisionConfig

LAYER_TYPES = ("linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp")


class Nemotron4TextConfig(PretrainedConfig):
    """Config of the Nemotron4 language model.

    Depth comes from ``layer_types``, which names the block at each position:
    ``linear_attention`` (GDP), ``mlp``, and the two shortcut MoE blocks whose
    mixer is either attention (``shortcut_moe_attn``) or GDP
    (``shortcut_moe_gdp``).
    """

    model_type = "nemotron4_text"

    def __init__(self, layer_types: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.layer_types = list(layer_types or [])
        if unknown := set(self.layer_types) - set(LAYER_TYPES):
            raise ValueError(f"Unknown Nemotron4 layer types: {sorted(unknown)}")

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)


class Nemotron4Config(PretrainedConfig):
    model_type = "nemotron4"
    sub_configs = {
        "text_config": Nemotron4TextConfig,
        "vision_config": PixtralVisionConfig,
    }

    def __init__(
        self,
        text_config: dict | Nemotron4TextConfig | None = None,
        vision_config: dict | PixtralVisionConfig | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(text_config, Nemotron4TextConfig):
            text_config = Nemotron4TextConfig(**(text_config or {}))
        if not isinstance(vision_config, PixtralVisionConfig):
            vision_config = PixtralVisionConfig(**(vision_config or {}))
        self.text_config = text_config
        self.vision_config = vision_config


def register_nemotron4_config() -> None:
    """Teach ``AutoConfig`` about ``model_type: nemotron4``.

    Idempotent, so importing this module more than once in a worker is safe.
    """
    AutoConfig.register("nemotron4_text", Nemotron4TextConfig, exist_ok=True)
    AutoConfig.register("nemotron4", Nemotron4Config, exist_ok=True)
