import json
import random
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from typing import Any, Literal, cast, overload

from haiway import MISSING, Missing, as_list, ctx
from openai import AsyncStream, Omit, omit
from openai import RateLimitError as OpenAIRateLimitError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import Choice
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.shared_params.function_parameters import FunctionParameters

from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelOutputStream,
    ModelRateLimit,
    ModelReasoning,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
    record_model_invocation,
    record_usage_metrics,
)
from draive.multimodal import (
    ArtifactContent,
    MultimodalContent,
    MultimodalContentPart,
    TextContent,
)
from draive.resources import ResourceContent, ResourceReference
from draive.vllm.api import VLLMAPI
from draive.vllm.config import VLLMChatConfig
from draive.vllm.utils import unwrap_missing

__all__ = ("VLLMMessages",)


class VLLMMessages(VLLMAPI):
    async def completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: VLLMChatConfig | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("vllm.completions"):
            config = config or ctx.state(VLLMChatConfig)
            record_model_invocation(
                provider=f"vllm@{self._base_url}",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
            )
            ctx.record_debug(
                attributes={
                    "model.instructions": instructions,
                    "model.tools": [tool.name for tool in tools.specification],
                    "model.context": [element.to_str() for element in context],
                },
            )

            tool_choice: ChatCompletionToolChoiceOptionParam | Omit
            tools_list: Iterable[ChatCompletionToolParam] | Omit
            tool_choice, tools_list = _tools_as_tool_config(
                tools.specification,
                tool_selection=tools.selection,
            )

            # Start streaming request
            stream: AsyncStream[ChatCompletionChunk]
            try:
                stream = await self._client.chat.completions.create(
                    model=config.model,
                    messages=_context_messages(
                        instructions=instructions,
                        context=context,
                        vision_details=config.vision_details,
                    ),
                    temperature=unwrap_missing(config.temperature),
                    top_p=unwrap_missing(config.top_p),
                    frequency_penalty=unwrap_missing(config.frequency_penalty),
                    tools=tools_list,
                    tool_choice=tool_choice,
                    parallel_tool_calls=unwrap_missing(config.parallel_tool_calls)
                    if tools_list is not omit
                    else omit,
                    max_tokens=unwrap_missing(config.max_output_tokens),
                    response_format=_response_format(output),
                    seed=unwrap_missing(config.seed),
                    stop=as_list(cast(Iterable[str], config.stop_sequences))
                    if config.stop_sequences is not MISSING
                    else omit,
                    stream=True,
                )

            except OpenAIRateLimitError as exc:
                delay: float
                try:
                    if retry_after := exc.response.headers.get("Retry-After"):
                        delay = float(retry_after)

                    else:
                        delay = random.uniform(0.3, 3.0)  # nosec: B311

                except Exception:
                    delay = random.uniform(0.3, 3.0)  # nosec: B311

                ctx.record_warning(
                    event="model.rate_limit",
                    attributes={
                        "model.provider": f"vllm@{self._base_url}",
                        "model.name": config.model,
                        "retry_after": delay,
                    },
                )
                raise ModelRateLimit(
                    provider=f"vllm@{self._base_url}",
                    model=config.model,
                    retry_after=delay,
                ) from exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider=f"vllm@{self._base_url}",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            # Accumulate tool call deltas by index and emit complete calls after stream ends.
            tool_accumulator: MutableMapping[int, MutableMapping[str, str]] = {}
            latest_input_tokens: int | None = None
            latest_output_tokens: int | None = None
            try:
                async for chunk in stream:  # ChatCompletionChunk
                    if usage := getattr(chunk, "usage", None):
                        latest_input_tokens = usage.prompt_tokens
                        latest_output_tokens = usage.completion_tokens

                    if not chunk.choices:
                        continue  # allow usage-only chunks

                    choice: Choice = chunk.choices[0]

                    if choice.delta.content:
                        yield TextContent(text=choice.delta.content)

                    # Accumulate tool call parts
                    if choice.delta.tool_calls:
                        for call in choice.delta.tool_calls:
                            tool_state: MutableMapping[str, str] = tool_accumulator.setdefault(
                                call.index,
                                {"arguments": ""},
                            )
                            if call.id:
                                tool_state["id"] = call.id

                            if call.function:
                                if call.function.name:
                                    # name may stream in segments; append if partial
                                    tool_state["name"] = (
                                        tool_state.get("name", "") + call.function.name
                                    )

                                if call.function.arguments:
                                    tool_state["arguments"] = (
                                        tool_state["arguments"] + call.function.arguments
                                    )

                    if choice.finish_reason == "length":
                        raise ModelOutputLimit(
                            provider=f"vllm@{self._base_url}",
                            model=config.model,
                            max_output_tokens=(
                                cast(int, config.max_output_tokens)
                                if config.max_output_tokens is not MISSING
                                else 0
                            ),
                        )

                    if choice.finish_reason in (None, "stop", "tool_calls", "function_call"):
                        continue

                    raise ModelOutputFailed(
                        provider=f"vllm@{self._base_url}",
                        model=config.model,
                        reason=f"Unsupported finish reason: {choice.finish_reason}",
                    )

                record_usage_metrics(
                    provider=f"vllm@{self._base_url}",
                    model=config.model,
                    input_tokens=latest_input_tokens,
                    output_tokens=latest_output_tokens,
                )

                for index in sorted(tool_accumulator):
                    tool_state = tool_accumulator[index]
                    match tool_state:
                        case {
                            "id": str() as identifier,
                            "name": str() as name,
                            "arguments": str() as args,
                        }:
                            try:
                                arguments: Mapping[str, Any] = json.loads(args) if args else {}

                            except Exception as exc:
                                raise ModelOutputInvalid(
                                    provider=f"vllm@{self._base_url}",
                                    model=config.model,
                                    reason=(
                                        "Tool arguments decoding error - "
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                ) from exc

                            yield ModelToolRequest.of(
                                identifier,
                                tool=name,
                                arguments=arguments,
                            )

                        case _:
                            raise ModelOutputInvalid(
                                provider="vllm",
                                model=config.model,
                                reason="Invalid tool request",
                            )

            except OpenAIRateLimitError as exc:
                delay: float
                try:
                    if retry_after := exc.response.headers.get("Retry-After"):
                        delay = float(retry_after)

                    else:
                        delay = random.uniform(0.3, 3.0)  # nosec: B311

                except Exception:
                    delay = random.uniform(0.3, 3.0)  # nosec: B311

                ctx.record_warning(
                    event="model.rate_limit",
                    attributes={
                        "model.provider": f"vllm@{self._base_url}",
                        "model.name": config.model,
                        "retry_after": delay,
                    },
                )
                raise ModelRateLimit(
                    provider=f"vllm@{self._base_url}",
                    model=config.model,
                    retry_after=delay,
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider=f"vllm@{self._base_url}",
                    model=config.model,
                    reason=str(exc),
                ) from exc


def _context_messages(
    *,
    instructions: ModelInstructions,
    context: ModelContext,
    vision_details: Literal["auto", "low", "high"] | Missing,
) -> Iterable[ChatCompletionMessageParam]:
    yield ChatCompletionSystemMessageParam(
        role="system",
        content=instructions,
    )

    for element in context:
        if isinstance(element, ModelInput):
            yield ChatCompletionUserMessageParam(
                role="user",
                content=_content_parts(
                    element.content.parts,
                    vision_details=vision_details,
                ),
            )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            content: MutableSequence[ChatCompletionContentPartTextParam] = []
            tool_calls: MutableSequence[ChatCompletionMessageFunctionToolCallParam] = []
            for block in element.output:
                if isinstance(block, MultimodalContent):
                    content.extend(
                        _content_parts(
                            block.parts,
                            vision_details=vision_details,
                            text_only=True,
                        )
                    )

                elif isinstance(block, ModelReasoning):
                    continue  # skip reasoning blocks - not supported in this api

                else:
                    tool_calls.append(
                        ChatCompletionMessageFunctionToolCallParam(
                            id=block.identifier,
                            type="function",
                            function=Function(
                                name=block.tool,
                                arguments=json.dumps(block.arguments),
                            ),
                        )
                    )

            yield ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )


@overload
def _content_parts(
    parts: Iterable[MultimodalContentPart],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
    text_only: Literal[True],
) -> Iterable[ChatCompletionContentPartTextParam]: ...


@overload
def _content_parts(
    parts: Iterable[MultimodalContentPart],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
    text_only: Literal[False] = False,
) -> Iterable[ChatCompletionContentPartParam]: ...


def _content_parts(  # noqa: C901, PLR0912
    parts: Iterable[MultimodalContentPart],
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
    text_only: bool = False,
) -> Iterable[ChatCompletionContentPartParam]:
    for part in parts:
        if isinstance(part, TextContent):
            yield ChatCompletionContentPartTextParam(
                type="text",
                text=part.text,
            )

        elif isinstance(part, ResourceReference):
            if text_only:
                continue  # skip with text only

            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            if vision_details is MISSING:
                yield ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=part.uri,
                    ),
                )

            else:
                yield ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=part.uri,
                        detail=cast(Literal["auto", "low", "high"], vision_details),
                    ),
                )

        elif isinstance(part, ResourceContent):
            if text_only:
                continue  # skip with text only

            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            if vision_details is MISSING:
                yield ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=part.to_data_uri(),
                    ),
                )

            else:
                yield ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=part.to_data_uri(),
                        detail=cast(Literal["auto", "low", "high"], vision_details),
                    ),
                )

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden artifacts

            yield ChatCompletionContentPartTextParam(
                type="text",
                text=part.to_str(),
            )


def _tools_as_tool_config(
    tools: Sequence[ModelToolSpecification],
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> tuple[
    ChatCompletionToolChoiceOptionParam | Omit,
    Iterable[ChatCompletionToolParam] | Omit,
]:
    if not tools:
        return (omit, omit)

    tools_list: list[ChatCompletionToolParam] = [
        ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description or "",
                parameters=cast(FunctionParameters, tool.parameters)
                or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        )
        for tool in tools
    ]

    if tool_selection == "auto":
        return ("auto", tools_list)

    if tool_selection == "none":
        return ("none", tools_list)

    if tool_selection == "required":
        return ("required", tools_list)

    return (  # specific tool name
        ChatCompletionNamedToolChoiceParam(
            type="function",
            function={"name": tool_selection.name},
        ),
        tools_list,
    )


def _response_format(
    output: ModelOutputSelection,
    /,
) -> ResponseFormat | Omit:
    if output == "json":
        return {"type": "json_object"}

    elif isinstance(output, type):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output.__name__,
                "schema": cast(dict[str, object], output.__SPECIFICATION__),
                "strict": False,
            },
        }

    else:
        return omit
