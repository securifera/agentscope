# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches
"""OpenAI Chat model class."""
import warnings
from datetime import datetime
from typing import (
    Any,
    TYPE_CHECKING,
    List,
    AsyncGenerator,
    Literal,
    Type,
)
from collections import OrderedDict

from pydantic import BaseModel, create_model

from . import ChatResponse
from ._model_base import ChatModelBase
from ._model_usage import ChatUsage
from .._logging import logger
from .._utils._common import _json_loads_with_repair
from ..message import (
    ToolUseBlock,
    TextBlock,
    ThinkingBlock,
    AudioBlock,
    Base64Source,
)
from ..tracing import trace_llm
from ..types import JSONSerializableObject

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai import AsyncStream
else:
    ChatCompletion = "openai.types.chat.ChatCompletion"
    AsyncStream = "openai.types.chat.AsyncStream"

from openai import pydantic_function_tool


def _pydantic_model_from_schema(name: str, schema: dict) -> type[BaseModel]:
    fields = {}
    required = set(schema.get("required", []))

    for field, spec in schema.get("properties", {}).items():
        py_type = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }.get(spec.get("type"), Any)

        default = ... if field in required else spec.get("default", None)
        fields[field] = (py_type, default)

    return create_model(name, **fields)


def _convert_tools_for_responses_api(client, tools: list[dict], cache: dict | None = None) -> list[Any]:
    converted = []
    cache = cache or {}

    for t in tools:
        # Leave non-function tools alone (e.g. web_search)
        if t.get("type") != "function":
            converted.append(t)
            continue

        fn = t.get("function", t)

        # Already a Pydantic tool → keep
        if not isinstance(fn, dict):
            converted.append(t)
            continue

        # Check cache first
        fn_name = fn["name"]
        if fn_name in cache:
            converted.append(cache[fn_name])
            continue

        # Create and cache the Pydantic tool
        model = _pydantic_model_from_schema(
            f"{fn_name.title()}Args",
            fn["parameters"],
        )

        pydantic_tool = pydantic_function_tool(
            model,
            name=fn_name,
            description=fn.get("description", ""),
        )
        cache[fn_name] = pydantic_tool
        converted.append(pydantic_tool)

    return converted


def _format_audio_data_for_qwen_omni(messages: list[dict]) -> None:
    """Qwen-omni uses OpenAI-compatible API but requires different audio
    data format than OpenAI with "data:;base64," prefix.
    Refer to `Qwen-omni documentation
    <https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2867839>`_
    for more details.

    Args:
        messages (`list[dict]`):
            The list of message dictionaries from OpenAI formatter.
    """
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if (
                    isinstance(block, dict)
                    and "input_audio" in block
                    and isinstance(block["input_audio"].get("data"), str)
                ):
                    if not block["input_audio"]["data"].startswith("http"):
                        block["input_audio"]["data"] = (
                            "data:;base64," + block["input_audio"]["data"]
                        )


class OpenAIChatModel(ChatModelBase):
    """The OpenAI chat model class."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        stream: bool = True,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        organization: str = None,
        client_type: Literal["openai", "azure"] = "openai",
        client_kwargs: dict[str, JSONSerializableObject] | None = None,
        generate_kwargs: dict[str, JSONSerializableObject] | None = None,
        enable_web_search: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the openai client.

        Args:
            model_name (`str`, default `None`):
                The name of the model to use in OpenAI API.
            api_key (`str`, default `None`):
                The API key for OpenAI API. If not specified, it will
                be read from the environment variable `OPENAI_API_KEY`.
            stream (`bool`, default `True`):
                Whether to use streaming output or not.
            reasoning_effort (`Literal["low", "medium", "high"] | None`, \
            optional):
                Reasoning effort, supported for o3, o4, etc. Please refer to
                `OpenAI documentation
                <https://platform.openai.com/docs/guides/reasoning?api-mode=chat>`_
                for more details.
            organization (`str`, default `None`):
                The organization ID for OpenAI API. If not specified, it will
                be read from the environment variable `OPENAI_ORGANIZATION`.
            client_type (`Literal["openai", "azure"]`, default `openai`):
                Selects which OpenAI-compatible client to initialize.
            client_kwargs (`dict[str, JSONSerializableObject] | None`, \
             optional):
                The extra keyword arguments to initialize the OpenAI client.
            generate_kwargs (`dict[str, JSONSerializableObject] | None`, \
             optional):
                The extra keyword arguments used in OpenAI API generation,
                e.g. `temperature`, `seed`.
            enable_web_search (`bool`, default `True`):
                Whether to enable OpenAI's builtin web search tool. When True
                and using the newer Responses API, `{ "type": "web_search" }`
                will be included automatically. For legacy Chat Completions the
                tool type is not supported and will be omitted.
            **kwargs (`Any`):
                Additional keyword arguments.
        """

        # Handle deprecated client_args parameter from kwargs
        client_args = kwargs.pop("client_args", None)
        if client_args is not None and client_kwargs is not None:
            raise ValueError(
                "Cannot specify both 'client_args' and 'client_kwargs'. "
                "Please use only 'client_kwargs' (client_args is deprecated).",
            )

        if client_args is not None:
            logger.warning(
                "The parameter 'client_args' is deprecated and will be "
                "removed in a future version. Please use 'client_kwargs' "
                "instead. Automatically converting 'client_args' to "
                "'client_kwargs'.",
            )
            client_kwargs = client_args

        if kwargs:
            logger.warning(
                "Unknown keyword arguments: %s. These will be ignored.",
                list(kwargs.keys()),
            )

        super().__init__(model_name, stream)

        import openai

        if client_type not in ("openai", "azure"):
            raise ValueError(
                "Invalid client_type. Supported values: 'openai', 'azure'.",
            )

        if client_type == "azure":
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                organization=organization,
                **(client_kwargs or {}),
            )
        else:
            self.client = openai.AsyncClient(
                api_key=api_key,
                organization=organization,
                **(client_kwargs or {}),
            )

        self.reasoning_effort = reasoning_effort
        self.generate_kwargs = generate_kwargs or {}
        self.enable_web_search = enable_web_search
        self._pydantic_tools_cache: dict[str,
                                         Any] = {}  # Cache converted tools

    @trace_llm
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str | None = None,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """Get the response from OpenAI chat completions API by the given
        arguments.

        Args:
            messages (`list[dict]`):
                A list of dictionaries, where `role` and `content` fields are
                required, and `name` field is optional.
            tools (`list[dict]`, default `None`):
                The tools JSON schemas that the model can use.
            tool_choice (`Literal["auto", "none", "required"] | str \
            | None`, default `None`):
                Controls which (if any) tool is called by the model.
                 Can be "auto", "none", "required", or specific tool
                 name. For more details, please refer to
                 https://platform.openai.com/docs/api-reference/responses/create#responses_create-tool_choice
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output. When provided, the model will be forced
                to return data that conforms to this schema by automatically
                converting the BaseModel to a tool function and setting
                `tool_choice` to enforce its usage. This enables structured
                output generation.

                .. note:: When `structured_model` is specified,
                    both `tools` and `tool_choice` parameters are ignored,
                    and the model will only perform structured output
                    generation without calling any other tools.

                For more details, please refer to the `official document
                <https://platform.openai.com/docs/guides/structured-outputs>`_

            **kwargs (`Any`):
                The keyword arguments for OpenAI chat completions API,
                e.g. `temperature`, `max_tokens`, `top_p`, etc. Please
                refer to the OpenAI API documentation for more details.

        Returns:
            `ChatResponse | AsyncGenerator[ChatResponse, None]`:
                The response from the OpenAI chat completions API.
        """

        # checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )

        # Qwen-omni requires different base64 audio format from openai
        if "omni" in self.model_name.lower():
            _format_audio_data_for_qwen_omni(messages)

        # Decide whether to use Responses API (only when web_search tool is explicitly requested)
        has_web_search_tool = (
            tools and any(
                (isinstance(t, dict) and t.get("type") == "web_search")
                for t in tools
            )
        )
        use_responses_api = (
            self.enable_web_search
            and has_web_search_tool
            and not structured_model
            and hasattr(self.client, "responses")
        )

        start_datetime = datetime.now()

        if use_responses_api:
            try:
                # Convert chat messages to responses input blocks
                # The Responses API uses different block types than Chat Completions:
                # - Input: 'input_text', 'input_image', etc.
                # - Output: 'output_text', 'refusal', etc.
                responses_input = []
                for m in messages:
                    # Normalize role to Responses API supported values
                    raw_role = m.get("role", "user")
                    if raw_role is None:
                        raw_role = "user"
                    role = str(raw_role).lower()
                    if role not in ("assistant", "system", "developer", "user"):
                        # Map common legacy/custom roles to a supported role
                        if role in ("tool", "function", "tool_use", "tool-use", "agent"):
                            role = "assistant"
                        else:
                            role = "user"

                    if isinstance(m.get("content"), list):
                        # Content is structured blocks - need to convert types
                        content_items = []
                        for block in m["content"]:
                            block_copy = {**block}
                            # Convert legacy 'text' blocks to Responses API types
                            if block.get("type") == "text":
                                if role == "assistant":
                                    block_copy["type"] = "output_text"
                                else:
                                    block_copy["type"] = "input_text"
                                # Ensure 'text' is never None (Responses requires string)
                                if block_copy.get("text") is None:
                                    block_copy["text"] = ""
                            elif block.get("type") == "image_url":
                                block_copy["type"] = "input_image"
                                # Responses API expects 'source' not 'image_url'
                                if "image_url" in block_copy:
                                    block_copy["source"] = block_copy.pop(
                                        "image_url")
                            else:
                                # Defensive: make sure text fields are strings
                                if "text" in block_copy and block_copy["text"] is None:
                                    block_copy["text"] = ""
                            content_items.append(block_copy)
                    else:
                        # Simple string content - wrap in input_text or
                        # output_text block depending on the role. Coerce
                        # None -> empty string and non-string -> str().
                        content_type = "output_text" if role == "assistant" else "input_text"
                        raw = m.get("content")
                        if raw is None:
                            content_text = ""
                        elif isinstance(raw, str):
                            content_text = raw
                        else:
                            content_text = str(raw)
                        content_items = [
                            {"type": content_type, "text": content_text}]

                    responses_input.append(
                        {
                            "role": role,
                            "content": content_items,
                        },
                    )

                tools_to_send: list[Any] = []
                if tools:
                    tools_to_send.extend(tools)

                # Add web_search tool if enabled and not already present
                if self.enable_web_search:
                    if not any(
                        (isinstance(t, dict) and t.get("type") == "web_search")
                        for t in tools_to_send
                    ):
                        tools_to_send.append({"type": "web_search"})

                # Convert legacy function tools → Pydantic tools (with caching)
                tools_to_send = _convert_tools_for_responses_api(
                    self.client,
                    tools_to_send,
                    cache=self._pydantic_tools_cache,)

                resp_kwargs = {
                    "model": self.model_name,
                    "input": responses_input,
                    **self.generate_kwargs,
                    **kwargs,
                }
                if self.reasoning_effort and "reasoning_effort" not in resp_kwargs:
                    resp_kwargs["reasoning_effort"] = self.reasoning_effort
                if tools_to_send:
                    resp_kwargs["tools"] = tools_to_send
                if tool_choice:
                    self._validate_tool_choice(tool_choice, tools_to_send)
                    resp_kwargs["tool_choice"] = self._format_tool_choice(
                        tool_choice,
                        responses_api=True,
                    )

                logger.debug(
                    "Using Responses API (web_search enabled) model=%s", self.model_name)
                if self.stream:
                    # responses.stream returns a manager directly (not awaited)
                    stream = self.client.responses.stream(**resp_kwargs)
                    return self._parse_openai_responses_stream(
                        start_datetime,
                        stream,
                    )
                else:
                    resp = await self.client.responses.create(**resp_kwargs)
                    return self._parse_openai_responses_response(
                        start_datetime,
                        resp,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Responses API failed (%s). Falling back to chat completions.",
                    e,
                )
                # fall through to chat completions with original messages format

        # Legacy Chat Completions path
        logger.debug("Using Chat Completions API model=%s", self.model_name)
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": self.stream,
            **self.generate_kwargs,
            **kwargs,
        }
        if self.reasoning_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort

        tools_to_send: list[dict] = []
        if tools:
            tools_to_send.extend(tools)
        # Do NOT append web_search here (unsupported in chat completions)
        if tools_to_send:
            kwargs["tools"] = self._format_tools_json_schemas(tools_to_send)
        if tool_choice:
            # Handle deprecated "any" option with warning
            if tool_choice == "any":
                warnings.warn(
                    '"any" is deprecated and will be removed in a future '
                    "version.",
                    DeprecationWarning,
                )
                tool_choice = "required"
            self._validate_tool_choice(tool_choice, tools)
            kwargs["tool_choice"] = self._format_tool_choice(
                tool_choice, responses_api=False,)

        if self.stream:
            kwargs["stream_options"] = {"include_usage": True}

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and 'tool_choice' will be ignored.",
                )
            kwargs.pop("stream", None)
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            kwargs["response_format"] = structured_model
            if not self.stream:
                response = await self.client.chat.completions.parse(**kwargs)
            else:
                response = self.client.chat.completions.stream(**kwargs)
                return self._parse_openai_stream_response(
                    start_datetime,
                    response,
                    structured_model,
                )
        else:
            response = await self.client.chat.completions.create(**kwargs)

        if self.stream:
            return self._parse_openai_stream_response(
                start_datetime,
                response,
                structured_model,
            )
        return self._parse_openai_completion_response(
            start_datetime,
            response,
            structured_model,
        )

    async def _parse_openai_stream_response(
        self,
        start_datetime: datetime,
        response: AsyncStream,
        structured_model: Type[BaseModel] | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Given an OpenAI streaming completion response, extract the content
         blocks and usages from it and yield ChatResponse objects.

        Args:
            start_datetime (`datetime`):
                The start datetime of the response generation.
            response (`AsyncStream`):
                OpenAI AsyncStream object to parse.
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output.

        Returns:
            `AsyncGenerator[ChatResponse, None]`:
                An async generator that yields ChatResponse objects containing
                the content blocks and usage information for each chunk in
                the streaming response.

        .. note::
            If `structured_model` is not `None`, the expected structured output
            will be stored in the metadata of the `ChatResponse`.
        """
        usage, res = None, None
        text = ""
        thinking = ""
        audio = ""
        tool_calls = OrderedDict()
        metadata: dict | None = None
        contents: List[
            TextBlock | ToolUseBlock | ThinkingBlock | AudioBlock
        ] = []

        async with response as stream:
            async for item in stream:
                if structured_model:
                    if item.type != "chunk":
                        continue
                    chunk = item.chunk
                else:
                    chunk = item

                if chunk.usage:
                    usage = ChatUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        time=(datetime.now() - start_datetime).total_seconds(),
                    )

                if not chunk.choices:
                    if usage and contents:
                        res = ChatResponse(
                            content=contents,
                            usage=usage,
                            metadata=metadata,
                        )
                        yield res
                    continue

                choice = chunk.choices[0]

                thinking += (
                    getattr(choice.delta, "reasoning_content", None) or ""
                )
                text += getattr(choice.delta, "content", None) or ""

                if (
                    hasattr(choice.delta, "audio")
                    and "data" in choice.delta.audio
                ):
                    audio += choice.delta.audio["data"]
                if (
                    hasattr(choice.delta, "audio")
                    and "transcript" in choice.delta.audio
                ):
                    text += choice.delta.audio["transcript"]

                for tool_call in choice.delta.tool_calls or []:
                    if tool_call.index in tool_calls:
                        if tool_call.function.arguments is not None:
                            tool_calls[tool_call.index][
                                "input"
                            ] += tool_call.function.arguments

                    else:
                        tool_calls[tool_call.index] = {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": tool_call.function.arguments or "",
                        }

                contents = []

                if thinking:
                    contents.append(
                        ThinkingBlock(
                            type="thinking",
                            thinking=thinking,
                        ),
                    )

                if audio:
                    media_type = self.generate_kwargs.get("audio", {}).get(
                        "format",
                        "wav",
                    )
                    contents.append(
                        AudioBlock(
                            type="audio",
                            source=Base64Source(
                                data=audio,
                                media_type=f"audio/{media_type}",
                                type="base64",
                            ),
                        ),
                    )

                if text:
                    contents.append(
                        TextBlock(
                            type="text",
                            text=text,
                        ),
                    )

                    if structured_model:
                        metadata = _json_loads_with_repair(text)

                for tool_call in tool_calls.values():
                    contents.append(
                        ToolUseBlock(
                            type=tool_call["type"],
                            id=tool_call["id"],
                            name=tool_call["name"],
                            input=_json_loads_with_repair(
                                tool_call["input"] or "{}",
                            ),
                        ),
                    )

                if not contents:
                    continue

                res = ChatResponse(
                    content=contents,
                    usage=usage,
                    metadata=metadata,
                )
                yield res

    def _parse_openai_completion_response(
        self,
        start_datetime: datetime,
        response: ChatCompletion,
        structured_model: Type[BaseModel] | None = None,
    ) -> ChatResponse:
        """Given an OpenAI chat completion response object, extract the content
            blocks and usages from it.

        Args:
            start_datetime (`datetime`):
                The start datetime of the response generation.
            response (`ChatCompletion`):
                OpenAI ChatCompletion object to parse.
            structured_model (`Type[BaseModel] | None`, default `None`):
                A Pydantic BaseModel class that defines the expected structure
                for the model's output.

        Returns:
            ChatResponse (`ChatResponse`):
                A ChatResponse object containing the content blocks and usage.

        .. note::
            If `structured_model` is not `None`, the expected structured output
            will be stored in the metadata of the `ChatResponse`.
        """
        content_blocks: List[
            TextBlock | ToolUseBlock | ThinkingBlock | AudioBlock
        ] = []
        metadata: dict | None = None

        if response.choices:
            choice = response.choices[0]
            if (
                hasattr(choice.message, "reasoning_content")
                and choice.message.reasoning_content is not None
            ):
                content_blocks.append(
                    ThinkingBlock(
                        type="thinking",
                        thinking=response.choices[0].message.reasoning_content,
                    ),
                )

            if choice.message.content:
                content_blocks.append(
                    TextBlock(
                        type="text",
                        text=response.choices[0].message.content,
                    ),
                )
            if choice.message.audio:
                media_type = self.generate_kwargs.get("audio", {}).get(
                    "format",
                    "mp3",
                )
                content_blocks.append(
                    AudioBlock(
                        type="audio",
                        source=Base64Source(
                            data=choice.message.audio.data,
                            media_type=f"audio/{media_type}",
                            type="base64",
                        ),
                    ),
                )

                if choice.message.audio.transcript:
                    content_blocks.append(
                        TextBlock(
                            type="text",
                            text=choice.message.audio.transcript,
                        ),
                    )

            for tool_call in choice.message.tool_calls or []:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=_json_loads_with_repair(
                            tool_call.function.arguments,
                        ),
                    ),
                )

            if structured_model:
                metadata = choice.message.parsed.model_dump()

        usage = None
        if response.usage:
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                time=(datetime.now() - start_datetime).total_seconds(),
            )

        parsed_response = ChatResponse(
            content=content_blocks,
            usage=usage,
            metadata=metadata,
        )

        return parsed_response

    def _format_tools_json_schemas(
        self,
        schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format the tools JSON schemas to the OpenAI format."""
        return schemas

    def _format_tool_choice(
        self,
        tool_choice: Literal["auto", "none", "any", "required"] | str | None,
        responses_api: bool = False,
    ) -> str | dict | None:
        """Format tool_choice parameter for API compatibility.

        Args:
            tool_choice (`Literal["auto", "none", "required"] | str \
            | None`, default `None`):
                Controls which (if any) tool is called by the model.
                 Can be "auto", "none", "required", or specific tool name.
                 For more details, please refer to
                 https://platform.openai.com/docs/api-reference/responses/create#responses_create-tool_choice
        Returns:
            `dict | None`:
                The formatted tool choice configuration dict, or None if
                    tool_choice is None.
        """
        if tool_choice is None:
            return None

        mode_mapping = {
            "auto": "auto",
            "none": "none",
            "required": "required",
        }
        if tool_choice in mode_mapping:
            return mode_mapping[tool_choice]
        if isinstance(tool_choice, str):
            if tool_choice == "web_search" and responses_api:
                return {"type": "web_search"}
            return {"type": "function", "function": {"name": tool_choice}}
        return None

    async def _parse_openai_responses_stream(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Parse streaming Responses API output into ChatResponse objects."""
        text = ""
        audio = ""
        tool_calls = OrderedDict()
        usage = None
        async with response as stream:
            async for event in stream:
                # logger.debug("Responses stream event: %s", str(event))
                et = getattr(event, "type", None)
                if et == "response.error":
                    logger.error("Responses stream error: %s",
                                 getattr(event, "error", None))
                    continue
                if et == "response.completed":
                    resp = getattr(event, "response", None)
                    if resp and getattr(resp, "usage", None):
                        u = resp.usage
                        # ResponseUsage uses input_text_tokens/output_text_tokens
                        input_tokens = (
                            getattr(u, "input_text_tokens", None)
                            or getattr(u, "prompt_tokens", None)
                            or 0
                        )
                        output_tokens = (
                            getattr(u, "output_text_tokens", None)
                            or getattr(u, "completion_tokens", None)
                            or 0
                        )
                        usage = ChatUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            time=(datetime.now() -
                                  start_datetime).total_seconds(),
                        )
                if et == "response.output_text.delta":
                    text += event.delta
                if et == "response.output_audio.delta":
                    audio += event.delta
                if et == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", None) == "function_call":
                        tool_calls[item.id] = {
                            "type": "tool_use",
                            "id": item.call_id,
                            "name": item.name,
                            "input": "",
                        }

                if et == "response.function_call_arguments.delta":
                    tc = tool_calls.get(event.item_id)
                    if tc:
                        tc["input"] += event.delta

                # if et == "response.output_tool_calls.delta":
                #     for tc in event.delta:
                #         idx = tc.get("index")
                #         fn = tc.get("function", {})
                #         args = fn.get("arguments")
                #         if idx not in tool_calls:
                #             tool_calls[idx] = {
                #                 "type": "tool_use",
                #                 "id": tc.get("id"),
                #                 "name": fn.get("name"),
                #                 "input": args if isinstance(args, dict) else "",
                #             }
                #         else:
                #             if isinstance(args, str):
                #                 tool_calls[idx]["input"] += args
                #             elif isinstance(args, dict):
                #                 tool_calls[idx]["input"] = args

                if et and et.startswith("response."):
                    contents: list[Any] = []
                    if audio:
                        contents.append(
                            AudioBlock(
                                type="audio",
                                source=Base64Source(
                                    data=audio,
                                    media_type="audio/wav",
                                    type="base64",
                                ),
                            ),
                        )
                    if text:
                        contents.append(TextBlock(type="text", text=text))
                    for tc in tool_calls.values():
                        contents.append(
                            ToolUseBlock(
                                type="tool_use",
                                id=tc["id"],
                                name=tc["name"],
                                input=(
                                    tc["input"]
                                    if isinstance(tc["input"], dict)
                                    else _json_loads_with_repair(tc["input"] or "{}")
                                ),
                            ),
                        )
                    if not contents:
                        continue
                    yield ChatResponse(content=contents, usage=usage)

    def _parse_openai_responses_response(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> ChatResponse:
        """Parse non-streaming Responses API output into ChatResponse."""
        contents: list[Any] = []
        usage = None
        output = getattr(response, "output", None)
        if output:
            for item in output:
                if item.type == "output_text":
                    contents.append(TextBlock(type="text", text=item.text))
                if item.type == "output_audio":
                    contents.append(
                        AudioBlock(
                            type="audio",
                            source=Base64Source(
                                data=item.audio.get("data", ""),
                                media_type="audio/wav",
                                type="base64",
                            ),
                        ),
                    )
                # if item.type == "output_tool_calls":
                #     for tc in item.tool_calls:
                #         fn = tc.get("function", {})
                #         args = fn.get("arguments", {})
                #         contents.append(
                #             ToolUseBlock(
                #                 type="tool_use",
                #                 id=tc.get("id"),
                #                 name=fn.get("name"),
                #                 input=args if isinstance(args, dict)
                #                 else _json_loads_with_repair(args),
                #             ),
                #         )
                if item.type == "function_call":
                    contents.append(
                        ToolUseBlock(
                            type="tool_use",
                            id=item.call_id,
                            name=item.name,
                            input=item.parsed_arguments
                            if hasattr(item, "parsed_arguments")
                            else _json_loads_with_repair(item.arguments or "{}"),
                        )
                    )

        if getattr(response, "usage", None):
            u = response.usage
            # ResponseUsage uses input_text_tokens/output_text_tokens
            input_tokens = (
                getattr(u, "input_text_tokens", None)
                or getattr(u, "prompt_tokens", None)
                or 0
            )
            output_tokens = (
                getattr(u, "output_text_tokens", None)
                or getattr(u, "completion_tokens", None)
                or 0
            )
            usage = ChatUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                time=(datetime.now() - start_datetime).total_seconds(),
            )
        return ChatResponse(content=contents, usage=usage)
