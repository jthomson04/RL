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

from types import SimpleNamespace

import pytest

from nemo_rl.utils.prefix_reuse import (
    derive_required_prefix_token_ids,
    messages_to_last_assistant,
    replace_prefix_tokens,
)


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids):
        return repr(token_ids)


def test_replace_prefix_tokens_preserves_prior_model_tokens() -> None:
    result = replace_prefix_tokens(
        _Tokenizer(),
        model_prefix_token_ids=[11, 12, 220, 17, 2],
        template_prefix_token_ids=[11, 12, 1001, 2],
        template_token_ids=[11, 12, 1001, 2, 21, 22],
    )

    assert result == [11, 12, 220, 17, 2, 21, 22]


def test_replace_prefix_tokens_supports_shorter_full_template() -> None:
    result = replace_prefix_tokens(
        _Tokenizer(),
        model_prefix_token_ids=[11, 12, 220, 17, 2],
        template_prefix_token_ids=[9, 2, 11, 12, 1001, 1002, 1003, 2],
        template_token_ids=[9, 2, 11, 12, 2, 21, 22],
    )

    assert result == [11, 12, 220, 17, 2, 21, 22]


def test_replace_prefix_tokens_ignores_later_eos_after_matching_prefix() -> None:
    result = replace_prefix_tokens(
        _Tokenizer(),
        model_prefix_token_ids=[100, 2],
        template_prefix_token_ids=[9, 2],
        template_token_ids=[9, 2, 9, 2, 77, 88],
    )

    assert result == [100, 2, 9, 2, 77, 88]


def test_replace_prefix_tokens_finds_eos_after_old_prefix_bound() -> None:
    result = replace_prefix_tokens(
        _Tokenizer(),
        model_prefix_token_ids=[11, 12, 220, 17, 2],
        template_prefix_token_ids=[11, 12],
        template_token_ids=[11, 12, 1001, 2, 21, 22],
    )

    assert result == [11, 12, 220, 17, 2, 21, 22]


def test_replace_prefix_tokens_missing_eos_in_full_template_raises() -> None:
    with pytest.raises(AssertionError, match="No EOS token ID found"):
        replace_prefix_tokens(
            _Tokenizer(),
            model_prefix_token_ids=[11, 12, 220, 17, 2],
            template_prefix_token_ids=[11, 12, 1001],
            template_token_ids=[11, 12, 1001, 21, 22],
        )


def test_derive_required_prefix_token_ids_uses_latest_message() -> None:
    messages = [
        {"role": "assistant", "prompt_token_ids": [1], "generation_token_ids": [2]},
        {"role": "user", "content": "next"},
        SimpleNamespace(
            role="assistant",
            prompt_token_ids=[3, 4],
            generation_token_ids=[5],
        ),
    ]

    assert derive_required_prefix_token_ids(messages) == [3, 4, 5]


def test_messages_to_last_assistant_includes_latest_assistant_turn() -> None:
    messages = [
        {"role": "system"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "after"},
    ]

    assert messages_to_last_assistant(messages) == messages[:4]
