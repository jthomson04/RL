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

"""OpenAI-compatible token wrapper for NeMo-Gym traffic to Dynamo."""

import asyncio
import json
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Optional

import orjson

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.openai_server_utils import replace_prefix_tokens

_GYM_TOKEN_METADATA_FIELDS = (
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
)
_TOOL_ARGUMENT_MAPPING_ERROR = "Can only get item pairs from a mapping."
_REQUEST_PROCESS_START_TIMEOUT_S = 60

_request_worker_tokenizer: Any = None
_request_worker_tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None
_request_worker_exclude_tools_when_tool_choice_none: Optional[bool] = None


def _coerce_token_id_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of token IDs.")
    if all(type(token_id) is int for token_id in value):
        return value
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must contain only integer token IDs.") from e


def _strip_gym_token_metadata(messages: list[Any]) -> list[Any]:
    stripped_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            stripped_messages.append(message)
            continue
        stripped_message = dict(message)
        for field in _GYM_TOKEN_METADATA_FIELDS:
            stripped_message.pop(field, None)
        stripped_messages.append(stripped_message)
    return stripped_messages


def _chat_template_kwargs(
    request_body: dict[str, Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if tokenizer_chat_template_kwargs is not None:
        if not isinstance(tokenizer_chat_template_kwargs, dict):
            raise ValueError("tokenizer chat_template_kwargs must be a JSON object.")
        kwargs.update(tokenizer_chat_template_kwargs)

    request_kwargs = request_body.get("chat_template_kwargs")
    if request_kwargs is not None:
        if not isinstance(request_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a JSON object.")
        kwargs.update(request_kwargs)

    if "reasoning_effort" in request_body:
        kwargs["reasoning_effort"] = request_body["reasoning_effort"]
    return kwargs


def _request_add_generation_prompt(request_body: dict[str, Any]) -> bool:
    if "add_generation_prompt" in request_body:
        return bool(request_body["add_generation_prompt"])
    return not bool(request_body.get("continue_final_message", False))


def _apply_chat_template(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    add_generation_prompt: bool,
    tokenize: bool,
) -> Any:
    tools = request_body.get("tools")
    if (
        exclude_tools_when_tool_choice_none
        and request_body.get("tool_choice") == "none"
    ):
        tools = None

    apply_chat_template = type(tokenizer).apply_chat_template
    return apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        documents=request_body.get("documents"),
        chat_template=request_body.get("chat_template"),
        add_generation_prompt=add_generation_prompt,
        continue_final_message=bool(request_body.get("continue_final_message", False)),
        tokenize=tokenize,
        return_tensors=None,
        return_dict=False,
        **_chat_template_kwargs(request_body, tokenizer_chat_template_kwargs),
    )


def _render_prompt_token_ids(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids = _apply_chat_template(
        tokenizer=tokenizer,
        request_body=request_body,
        messages=messages,
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )

    if isinstance(token_ids, list) and (
        not token_ids or not isinstance(token_ids[0], list)
    ):
        return token_ids
    if isinstance(token_ids, list) and len(token_ids) == 1:
        return token_ids[0]
    raise ValueError(
        "Dynamo token wrapper expected chat template rendering to return one "
        "list of prompt token IDs."
    )


def _render_prompt_token_ids_with_optional_prefix(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    add_generation_prompt: bool,
    assistant_index: int | None,
) -> tuple[list[int], list[int] | None]:
    full_prompt_token_ids = _render_prompt_token_ids(
        tokenizer=tokenizer,
        request_body=request_body,
        messages=messages,
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        add_generation_prompt=add_generation_prompt,
    )
    if assistant_index is None:
        return full_prompt_token_ids, None

    template_prefix_token_ids = _render_prompt_token_ids(
        tokenizer=tokenizer,
        request_body=request_body,
        messages=messages[: assistant_index + 1],
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
        add_generation_prompt=False,
    )
    return full_prompt_token_ids, template_prefix_token_ids


def _latest_tokenized_assistant_index(messages: list[Any]) -> Optional[int]:
    for index in reversed(range(len(messages))):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if (
            message.get("prompt_token_ids") is not None
            and message.get("generation_token_ids") is not None
        ):
            return index
    return None


def _normalize_tool_arguments_for_template(
    messages: list[Any], *, before_index: int
) -> None:
    """Make OpenAI tool calls renderable by model chat templates.

    OpenAI chat messages carry ``function.arguments`` as a JSON string, while
    some model templates iterate those arguments as a mapping. Normalize only
    the local template copy; the request forwarded to Dynamo retains its
    original OpenAI payload.
    """
    for message in messages[:before_index]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", tool_call)
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
            function["arguments"] = (
                parsed_arguments if isinstance(parsed_arguments, dict) else {}
            )


def prepare_dynamo_chat_completion_request(
    request_body: dict[str, Any],
    *,
    tokenizer: Any,
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None,
    exclude_tools_when_tool_choice_none: bool,
) -> dict[str, Any]:
    """Prepare a NeMo-Gym chat-completion request for Dynamo token input.

    Gym owns prefix derivation. A tokenized assistant message and its Gym prefix
    must be present together.
    """
    if request_body.get("stream"):
        raise ValueError("Dynamo native token wrapper does not support stream=True.")

    n = request_body.get("n", 1)
    if n is not None and int(n) != 1:
        raise ValueError("Dynamo native token wrapper currently supports only n=1.")

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Dynamo token wrapper requires chat-completion messages.")

    prepared_body = dict(request_body)
    stripped_messages = _strip_gym_token_metadata(messages)
    prepared_body["messages"] = stripped_messages
    required_prefix_value = prepared_body.pop("required_prefix_token_ids", None)
    required_prefix_token_ids = (
        _coerce_token_id_list(
            required_prefix_value,
            "required_prefix_token_ids",
        )
        if required_prefix_value is not None
        else None
    )

    add_generation_prompt = _request_add_generation_prompt(prepared_body)
    template_messages = stripped_messages
    assistant_index = _latest_tokenized_assistant_index(messages)
    if assistant_index is not None and not required_prefix_token_ids:
        raise ValueError(
            "A tokenized assistant message requires required_prefix_token_ids. "
            "Set return_token_id_information: true and request "
            'nvext.extra_fields=["engine_data"].'
        )
    if assistant_index is None and required_prefix_token_ids is not None:
        raise ValueError(
            "required_prefix_token_ids requires a tokenized assistant message."
        )

    try:
        (
            full_prompt_token_ids,
            template_prefix_token_ids,
        ) = _render_prompt_token_ids_with_optional_prefix(
            tokenizer=tokenizer,
            request_body=prepared_body,
            messages=template_messages,
            tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
            exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
            add_generation_prompt=add_generation_prompt,
            assistant_index=assistant_index,
        )
    except TypeError as e:
        if str(e) != _TOOL_ARGUMENT_MAPPING_ERROR:
            raise
        template_messages = deepcopy(stripped_messages)
        _normalize_tool_arguments_for_template(
            template_messages, before_index=len(template_messages)
        )
        (
            full_prompt_token_ids,
            template_prefix_token_ids,
        ) = _render_prompt_token_ids_with_optional_prefix(
            tokenizer=tokenizer,
            request_body=prepared_body,
            messages=template_messages,
            tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
            exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
            add_generation_prompt=add_generation_prompt,
            assistant_index=assistant_index,
        )

    if required_prefix_token_ids is not None:
        assert template_prefix_token_ids is not None
        full_prompt_token_ids = replace_prefix_tokens(
            tokenizer,
            model_prefix_token_ids=required_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=full_prompt_token_ids,
        )

    nvext = prepared_body.get("nvext")
    if nvext is None:
        nvext = {}
    if not isinstance(nvext, dict):
        raise ValueError("nvext must be a JSON object.")
    nvext = dict(nvext)
    extra_fields = nvext.get("extra_fields", [])
    if not isinstance(extra_fields, list):
        raise ValueError("nvext.extra_fields must be a JSON list.")
    nvext["extra_fields"] = list(dict.fromkeys([*extra_fields, "engine_data"]))
    nvext["token_data"] = full_prompt_token_ids
    prepared_body["nvext"] = nvext

    return prepared_body


def _prepare_serialized_dynamo_chat_completion_request(
    request_body: bytes,
    *,
    tokenizer: Any,
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
) -> bytes:
    """Parse, prepare, and serialize one Dynamo chat-completion request."""
    parsed_body = orjson.loads(request_body)
    if not isinstance(parsed_body, dict):
        raise ValueError("Chat completion body must be a JSON object.")
    prepared_body = prepare_dynamo_chat_completion_request(
        parsed_body,
        tokenizer=tokenizer,
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        exclude_tools_when_tool_choice_none=exclude_tools_when_tool_choice_none,
    )
    return orjson.dumps(prepared_body)


def _initialize_request_worker(
    tokenizer: Any,
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    exclude_tools_when_tool_choice_none: bool,
    startup_barrier: Any,
) -> None:
    """Initialize one spawned request-preparation worker."""
    global _request_worker_tokenizer
    global _request_worker_tokenizer_chat_template_kwargs
    global _request_worker_exclude_tools_when_tool_choice_none

    _request_worker_tokenizer = tokenizer
    _request_worker_tokenizer_chat_template_kwargs = tokenizer_chat_template_kwargs
    _request_worker_exclude_tools_when_tool_choice_none = (
        exclude_tools_when_tool_choice_none
    )
    startup_barrier.wait()


def _prepare_serialized_request_in_worker(request_body: bytes) -> bytes:
    """Prepare one request with process-local tokenizer state."""
    if _request_worker_tokenizer is None:
        raise RuntimeError("Dynamo token wrapper request worker is not initialized.")
    if _request_worker_exclude_tools_when_tool_choice_none is None:
        raise RuntimeError("Dynamo token wrapper request worker is not initialized.")
    return _prepare_serialized_dynamo_chat_completion_request(
        request_body,
        tokenizer=_request_worker_tokenizer,
        tokenizer_chat_template_kwargs=(
            _request_worker_tokenizer_chat_template_kwargs
        ),
        exclude_tools_when_tool_choice_none=(
            _request_worker_exclude_tools_when_tool_choice_none
        ),
    )


def _request_worker_ready() -> bool:
    """Return after the request worker initializer has completed."""
    if _request_worker_tokenizer is None:
        raise RuntimeError("Dynamo token wrapper request worker is not initialized.")
    return True


class DynamoTokenWrapperServer:
    """Small HTTP server that supplies tokenized chat prompts to Dynamo."""

    def __init__(
        self,
        *,
        dynamo_frontend_base_url: str,
        tokenizer: Any,
        tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
        exclude_tools_when_tool_choice_none: bool,
        request_timeout_s: Optional[float],
        request_processes: int,
    ) -> None:
        self.dynamo_frontend_base_url = dynamo_frontend_base_url
        self.tokenizer = tokenizer
        self.tokenizer_chat_template_kwargs = tokenizer_chat_template_kwargs
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.request_timeout_s = request_timeout_s
        self.request_processes = request_processes
        self.base_url: Optional[str] = None
        self.server: Any = None
        self.thread: Optional[threading.Thread] = None
        self._client_session: Any = None
        self._request_executor: Optional[ProcessPoolExecutor] = None
        self._active_request_preparations = 0

    def _start_request_executor(self) -> None:
        if self.request_processes == 1:
            return

        context = multiprocessing.get_context("spawn")
        startup_barrier = context.Barrier(
            self.request_processes,
            timeout=_REQUEST_PROCESS_START_TIMEOUT_S,
        )
        executor = ProcessPoolExecutor(
            max_workers=self.request_processes,
            mp_context=context,
            initializer=_initialize_request_worker,
            initargs=(
                self.tokenizer,
                self.tokenizer_chat_template_kwargs,
                self.exclude_tools_when_tool_choice_none,
                startup_barrier,
            ),
        )
        started = False
        try:
            futures = [
                executor.submit(_request_worker_ready)
                for _ in range(self.request_processes)
            ]
            for future in futures:
                future.result(timeout=_REQUEST_PROCESS_START_TIMEOUT_S)
            started = True
        finally:
            if not started:
                executor.shutdown(wait=True, cancel_futures=True)
        self._request_executor = executor

    async def _prepare_request_body(self, request_body: bytes) -> bytes:
        executor = self._request_executor
        use_in_process_path = executor is None or self._active_request_preparations == 0
        self._active_request_preparations += 1
        try:
            if use_in_process_path:
                return await asyncio.to_thread(
                    _prepare_serialized_dynamo_chat_completion_request,
                    request_body,
                    tokenizer=self.tokenizer,
                    tokenizer_chat_template_kwargs=(
                        self.tokenizer_chat_template_kwargs
                    ),
                    exclude_tools_when_tool_choice_none=(
                        self.exclude_tools_when_tool_choice_none
                    ),
                )
            return await asyncio.get_running_loop().run_in_executor(
                executor,
                _prepare_serialized_request_in_worker,
                request_body,
            )
        finally:
            self._active_request_preparations -= 1

    def start(self) -> str:
        """Start the wrapper in a background uvicorn thread."""
        import aiohttp
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import Response

        self._start_request_executor()

        @asynccontextmanager
        async def lifespan(_: FastAPI):
            timeout = (
                aiohttp.ClientTimeout(total=self.request_timeout_s)
                if self.request_timeout_s is not None
                else aiohttp.ClientTimeout(total=None)
            )
            async with aiohttp.ClientSession(
                timeout=timeout,
                # Explicit: the default TCPConnector(limit=100) would cap the whole
                # rollout path at 100 concurrent requests to the frontend.
                connector=aiohttp.TCPConnector(limit=0),
            ) as session:
                self._client_session = session
                try:
                    yield
                finally:
                    self._client_session = None

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {
                "status": "ok",
                "dynamo_frontend_base_url": self.dynamo_frontend_base_url,
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Response:
            try:
                prepared_body = await self._prepare_request_body(await request.body())
            except orjson.JSONDecodeError:
                return Response(
                    content=orjson.dumps({"detail": "Invalid JSON body."}),
                    status_code=400,
                    media_type="application/json",
                )
            except ValueError as e:
                return Response(
                    content=orjson.dumps({"detail": str(e)}),
                    status_code=400,
                    media_type="application/json",
                )

            status_code, response_body = await self._forward_chat_completion(
                prepared_body,
                authorization=request.headers.get("authorization"),
            )
            return Response(
                content=response_body,
                status_code=status_code,
                media_type="application/json",
            )

        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local(
            DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_HTTP_PORT_RANGE_HIGH,
        )
        self.base_url = f"http://{node_ip}:{free_port}/v1"

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
            timeout_keep_alive=120,
        )
        self.server = uvicorn.Server(config=config)
        self.thread = threading.Thread(
            target=self.server.run,
            name="dynamo-token-wrapper",
            daemon=True,
        )
        self.thread.start()
        return self.base_url

    async def _forward_chat_completion(
        self,
        request_body: bytes,
        *,
        authorization: Optional[str],
    ) -> tuple[int, bytes]:
        import aiohttp

        url = f"{self.dynamo_frontend_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        session = self._client_session
        if session is None:
            return 503, orjson.dumps(
                {"error": {"message": "Dynamo token wrapper is not ready."}}
            )
        try:
            async with session.post(
                url,
                data=request_body,
                headers=headers,
            ) as response:
                return response.status, await response.read()
        except asyncio.TimeoutError:
            return 504, orjson.dumps(
                {"error": {"message": f"Timed out forwarding to {url}."}}
            )
        except aiohttp.ClientError as e:
            return 502, orjson.dumps(
                {
                    "error": {
                        "message": (
                            f"Failed to forward request to {url}: "
                            f"{type(e).__name__}: {e}"
                        )
                    }
                }
            )

    def shutdown(self) -> None:
        """Stop the background uvicorn server."""
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=10)
        request_executor = self._request_executor
        self._request_executor = None
        if request_executor is not None:
            request_executor.shutdown(wait=True, cancel_futures=True)
