import json
import random
from collections.abc import AsyncGenerator, Coroutine, Generator, Iterable
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import META_EMPTY, MISSING, ObservabilityLevel, as_list, ctx
from openai import NOT_GIVEN, NotGiven
from openai import RateLimitError as OpenAIRateLimitError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.chat.completion_create_params import ResponseFormat

from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputSelection,
    ModelRateLimit,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolsDeclaration,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.multimodal import (
    MediaData,
    MediaReference,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMChatConfig
from draive.vllm.utils import unwrap_missing

__all__ = ("VLLMMessages",)

# Consistent randomized backoff window for rate limits (seconds)
RATE_LIMIT_RETRY_RANGE: tuple[float, float] = (0.3, 3.0)


class VLLMMessages(VLLMAPI):
    def generative_model(self) -> GenerativeModel:
        return GenerativeModel(generating=self.completion)

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: Literal[False] = False,
        config: VLLMChatConfig | None = None,
        **extra: Any,
    ) -> Coroutine[None, None, ModelOutput]: ...

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: Literal[True],
        config: VLLMChatConfig | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        stream: bool = False,
        config: VLLMChatConfig | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                tools=tools,
                context=context,
                output=output,
                config=config or ctx.state(VLLMChatConfig),
                **extra,
            )

        return self._completion(
            instructions=instructions,
            tools=tools,
            context=context,
            output=output,
            config=config or ctx.state(VLLMChatConfig),
            **extra,
        )

    async def _completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: VLLMChatConfig,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "model.provider": "vllm",
                    "model.name": config.model,
                    "model.instructions": instructions,
                    "model.tools": [tool["name"] for tool in tools.specifications],
                    "model.tool_selection": tools.selection,
                    "model.context": [element.to_str() for element in context],
                    "model.temperature": config.temperature,
                    "model.output": str(output),
                    "model.streaming": False,
                },
            )

            messages: list[ChatCompletionMessageParam]
            if instructions:
                messages = [
                    cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "system",
                            "content": instructions,
                        },
                    ),
                    *_context_messages(
                        context,
                        vision_details=config.vision_details,
                    ),
                ]

            else:
                messages = list(
                    _context_messages(
                        context,
                        vision_details=config.vision_details,
                    )
                )

            # Response format
            response_format: ResponseFormat | NotGiven
            if output == "json":
                response_format = {"type": "json_object"}

            elif isinstance(output, type):
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": output.__name__,
                        "schema": cast(dict[str, Any], output.__PARAMETERS_SPECIFICATION__),
                        "strict": False,
                    },
                }

            else:
                response_format = NOT_GIVEN

            tool_choice, tools_list = _tools_as_tool_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            # Execute chat completion
            try:
                completion: ChatCompletion = await self._client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    top_p=unwrap_missing(config.top_p),
                    frequency_penalty=unwrap_missing(config.frequency_penalty),
                    tools=tools_list,
                    tool_choice=tool_choice,
                    parallel_tool_calls=unwrap_missing(config.parallel_tool_calls)
                    if tools_list is not NOT_GIVEN
                    else NOT_GIVEN,
                    max_tokens=unwrap_missing(config.max_output_tokens),
                    response_format=response_format,
                    seed=unwrap_missing(config.seed),
                    stop=as_list(cast(Iterable[str], config.stop_sequences))
                    if config.stop_sequences is not MISSING
                    else NOT_GIVEN,
                    timeout=unwrap_missing(config.timeout),
                    stream=False,
                )

            except OpenAIRateLimitError as exc:
                delay: float
                try:
                    if retry_after := exc.response.headers.get("Retry-After"):
                        delay = float(retry_after)

                    else:
                        delay = random.uniform(*RATE_LIMIT_RETRY_RANGE)  # nosec: B311

                except Exception:
                    delay = random.uniform(*RATE_LIMIT_RETRY_RANGE)  # nosec: B311

                raise ModelRateLimit(
                    provider="vllm",
                    model=config.model,
                    retry_after=delay,
                ) from exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="vllm",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            if usage := completion.usage:
                ctx.record(
                    ObservabilityLevel.INFO,
                    metric="model.input_tokens",
                    value=usage.prompt_tokens,
                    unit="tokens",
                    kind="counter",
                    attributes={"model.provider": "vllm", "model.name": completion.model},
                )
                ctx.record(
                    ObservabilityLevel.INFO,
                    metric="model.output_tokens",
                    value=usage.completion_tokens,
                    unit="tokens",
                    kind="counter",
                    attributes={"model.provider": "vllm", "model.name": completion.model},
                )

            return ModelOutput.of(
                *_completion_as_output_content(completion),
                meta={
                    "identifier": completion.id,
                    "model": config.model,
                    "finish_reason": completion.choices[0].finish_reason,
                },
            )

    async def _completion_stream(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        context: ModelContext,
        output: ModelOutputSelection,
        config: VLLMChatConfig,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        ctx.record(
            ObservabilityLevel.INFO,
            attributes={
                "model.provider": "vllm",
                "model.name": config.model,
                "model.instructions": instructions,
                "model.tools": [tool["name"] for tool in tools.specifications],
                "model.tool_selection": tools.selection,
                "model.context": [element.to_str() for element in context],
                "model.temperature": config.temperature,
                "model.output": str(output),
                "model.streaming": True,
            },
        )

        messages: list[ChatCompletionMessageParam]
        if instructions:
            messages = [
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "system",
                        "content": instructions,
                    },
                ),
                *_context_messages(
                    context,
                    vision_details=config.vision_details,
                ),
            ]

        else:
            messages = list(
                _context_messages(
                    context,
                    vision_details=config.vision_details,
                )
            )

        response_format: ResponseFormat | NotGiven
        if output == "json":
            response_format = {"type": "json_object"}

        elif isinstance(output, type):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": output.__name__,
                    "schema": cast(dict[str, Any], output.__PARAMETERS_SPECIFICATION__),
                    "strict": False,
                },
            }

        else:
            response_format = NOT_GIVEN

        tool_choice, tools_list = _tools_as_tool_config(
            tools.specifications,
            tool_selection=tools.selection,
        )

        # Start streaming request
        try:
            stream = await self._client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                top_p=unwrap_missing(config.top_p),
                frequency_penalty=unwrap_missing(config.frequency_penalty),
                tools=tools_list,
                tool_choice=tool_choice,
                parallel_tool_calls=unwrap_missing(config.parallel_tool_calls)
                if tools_list is not NOT_GIVEN
                else NOT_GIVEN,
                max_tokens=unwrap_missing(config.max_output_tokens),
                response_format=response_format,
                seed=unwrap_missing(config.seed),
                stop=as_list(cast(Iterable[str], config.stop_sequences))
                if config.stop_sequences is not MISSING
                else NOT_GIVEN,
                timeout=unwrap_missing(config.timeout),
                stream=True,
            )
        except OpenAIRateLimitError as exc:
            delay: float
            try:
                if retry_after := exc.response.headers.get("Retry-After"):
                    delay = float(retry_after)

                else:
                    delay = random.uniform(*RATE_LIMIT_RETRY_RANGE)  # nosec: B311

            except Exception:
                delay = random.uniform(*RATE_LIMIT_RETRY_RANGE)  # nosec: B311

            raise ModelRateLimit(
                provider="vllm",
                model=config.model,
                retry_after=delay,
            ) from exc

        except Exception as exc:
            raise ModelOutputFailed(
                provider="vllm",
                model=config.model,
                reason=str(exc),
            ) from exc

        # Accumulate tool call deltas by index
        tool_accum: dict[int, dict[str, str]] = {}

        try:
            async for chunk in stream:  # ChatCompletionChunk
                choice: Choice = chunk.choices[0]

                if choice.delta.content:
                    yield TextContent(text=choice.delta.content)

                # Accumulate tool call parts
                if choice.delta.tool_calls:
                    for call in choice.delta.tool_calls:
                        idx = call.index
                        entry = tool_accum.setdefault(
                            idx,
                            {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            },
                        )

                        if call.id:
                            entry["id"] = call.id

                        if call.function:
                            if call.function.name:
                                # name may stream in segments; replace if provided
                                entry["name"] = call.function.name

                            if call.function.arguments:
                                entry["arguments"] += call.function.arguments

                # When tool calls are selected, emit requests
                if choice.finish_reason == "tool_calls" and tool_accum:
                    for entry in tool_accum.values():
                        name: str = entry.get("name") or "n/a"
                        identifier: str = entry.get("id") or uuid4().hex
                        args_text: str = entry.get("arguments") or ""
                        try:
                            arguments = json.loads(args_text) if args_text else {}

                        except Exception:
                            arguments = {}

                        yield ModelToolRequest.of(
                            identifier,
                            tool=name,
                            arguments=arguments,
                        )

                    tool_accum.clear()

        except OpenAIRateLimitError as exc:
            delay: float
            try:
                if retry_after := exc.response.headers.get("Retry-After"):
                    delay = float(retry_after)

                else:
                    delay = random.uniform(*RATE_LIMIT_RETRY_RANGE)  # nosec: B311

            except Exception:
                delay = random.uniform(*RATE_LIMIT_RETRY_RANGE)  # nosec: B311

            raise ModelRateLimit(
                provider="vllm",
                model=config.model,
                retry_after=delay,
            ) from exc

        except Exception as exc:
            raise ModelOutputFailed(
                provider="vllm",
                model=config.model,
                reason=str(exc),
            ) from exc


def _context_messages(
    context: ModelContext,
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Any,
) -> Iterable[ChatCompletionMessageParam]:
    for element in context:
        if isinstance(element, ModelInput):
            user_content = element.content
            yield cast(
                ChatCompletionMessageParam,
                {
                    "role": "user",
                    "content": list(
                        content_parts(
                            user_content.parts,
                            vision_details=vision_details,
                            text_only=False,
                        )
                    ),
                },
            )

        elif isinstance(element, ModelOutput):
            for block in element.blocks:
                if isinstance(block, MultimodalContent):
                    yield cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "assistant",
                            "content": list(
                                content_parts(
                                    block.parts,
                                    vision_details=vision_details,
                                    text_only=True,
                                )
                            ),
                        },
                    )

                elif isinstance(block, ModelReasoning):
                    continue  # skip reasoning

                else:
                    yield cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": block.identifier,
                                    "type": "function",
                                    "function": {
                                        "name": block.tool,
                                        "arguments": json.dumps(dict(block.arguments)),
                                    },
                                }
                            ],
                        },
                    )

        else:
            raise ValueError(f"Unsupported model context element: {type(element).__name__}")


# Inline helper formerly in draive.vllm.lmm
def content_parts(
    parts: Iterable[MultimodalContentElement],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Any,
    text_only: bool,
) -> Iterable[dict[str, Any]]:
    for element in parts:
        match element:
            case TextContent() as text:
                yield {
                    "type": "text",
                    "text": text.text,
                }

            case MediaReference() as ref if not text_only and ref.kind == "image":
                yield {
                    "type": "image_url",
                    "image_url": {
                        "url": ref.uri,
                    },
                    "detail": vision_details,
                }

            case MediaData() as data if not text_only and data.kind == "image":
                yield {
                    "type": "image_url",
                    "image_url": {
                        "url": data.to_data_uri(safe_encoding=False),
                    },
                    "detail": vision_details,
                }

            case other:
                yield {
                    "type": "text",
                    "text": other.to_json(),
                }


def _tool_specification_as_tool(tool: ModelToolSpecification) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"] or "",
            "parameters": cast(dict[str, Any], tool["parameters"])
            or {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _tools_as_tool_config(
    tools: Iterable[ModelToolSpecification] | None,
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> tuple[
    ChatCompletionToolChoiceOptionParam | NotGiven,
    list[ChatCompletionToolParam] | NotGiven,
]:
    tools_list: list[ChatCompletionToolParam] = [
        _tool_specification_as_tool(tool) for tool in (tools or [])
    ]
    if not tools_list:
        return (NOT_GIVEN, NOT_GIVEN)

    if tool_selection == "auto":
        return ("auto", tools_list)

    if tool_selection == "none":
        return ("none", tools_list)

    if tool_selection == "required":
        return ("required", tools_list)

    # specific tool name
    return (
        {
            "type": "function",
            "function": {
                "name": cast(str, tool_selection),
            },
        },
        tools_list,
    )


def _completion_as_output_content(
    completion: ChatCompletion,
) -> Generator[MultimodalContent | ModelToolRequest]:
    message: ChatCompletionMessage = completion.choices[0].message

    if message.content:
        yield MultimodalContent.of(TextContent(text=message.content))

    if message.tool_calls:
        for call in message.tool_calls:
            if call.type != "function" or not call.function:
                continue

            yield ModelToolRequest(
                identifier=call.id or uuid4().hex,
                tool=call.function.name,
                arguments=json.loads(call.function.arguments) if call.function.arguments else {},
                meta=META_EMPTY,
            )
