import json
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
    ChatCompletionToolMessageParam,
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
    ModelReasoning,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
    model_rate_limit,
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
        async with ctx.scope("model.invocation"):
            config = config or ctx.state(VLLMChatConfig)
            record_model_invocation(
                provider=self._provider,
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
                top_p=config.top_p,
                seed=config.seed,
                frequency_penalty=config.frequency_penalty,
                parallel_tool_calls=config.parallel_tool_calls,
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
                    # usage is only reported within the stream when explicitly requested
                    stream_options={"include_usage": True},
                )

            except OpenAIRateLimitError as exc:
                raise model_rate_limit(
                    provider=self._provider,
                    model=config.model,
                    retry_after=exc.response.headers.get("Retry-After"),
                ) from exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider=self._provider,
                    model=config.model,
                    reason=str(exc),
                ) from exc

            # Accumulate tool call deltas by index and emit complete calls after stream ends.
            tool_accumulator: MutableMapping[int, MutableMapping[str, str]] = {}
            latest_input_tokens: int | None = None
            latest_output_tokens: int | None = None
            try:
                async for chunk in stream:  # ChatCompletionChunk
                    if usage := chunk.usage:
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
                                    # the name streams in segments, yet some servers
                                    # repeat it whole within every fragment instead -
                                    # appending a repeat would duplicate the value
                                    accumulated_name: str = tool_state.get("name", "")
                                    if accumulated_name != call.function.name:
                                        tool_state["name"] = accumulated_name + call.function.name

                                if call.function.arguments:
                                    tool_state["arguments"] = (
                                        tool_state["arguments"] + call.function.arguments
                                    )

                    if choice.finish_reason == "length":
                        raise ModelOutputLimit(
                            provider=self._provider,
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
                        provider=self._provider,
                        model=config.model,
                        reason=f"Unsupported finish reason: {choice.finish_reason}",
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
                                    provider=self._provider,
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
                                provider=self._provider,
                                model=config.model,
                                reason="Invalid tool request",
                            )

            except OpenAIRateLimitError as exc:
                raise model_rate_limit(
                    provider=self._provider,
                    model=config.model,
                    retry_after=exc.response.headers.get("Retry-After"),
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider=self._provider,
                    model=config.model,
                    reason=str(exc),
                ) from exc

            finally:
                if latest_input_tokens is not None or latest_output_tokens is not None:
                    record_usage_metrics(
                        provider=self._provider,
                        model=config.model,
                        input_tokens=latest_input_tokens,
                        output_tokens=latest_output_tokens,
                    )

                # release the http stream, iteration may have ended
                # before the response was completed
                await stream.close()


def _context_messages(
    *,
    instructions: ModelInstructions,
    context: ModelContext,
    vision_details: Literal["auto", "low", "high"] | Missing,
) -> Iterable[ChatCompletionMessageParam]:
    if instructions:  # an empty system message is rejected by some servers
        yield ChatCompletionSystemMessageParam(
            role="system",
            content=instructions,
        )

    for element in context:
        if isinstance(element, ModelInput):
            if user_content := element.content:
                yield ChatCompletionUserMessageParam(
                    role="user",
                    content=_content_parts(
                        user_content.parts,
                        vision_details=vision_details,
                    ),
                )

            # provide tool responses as separate tool messages expected by the api
            for tool_response in element.tool_responses:
                yield ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_response.identifier,
                    content=_content_parts(
                        tool_response.content.parts,
                        vision_details=vision_details,
                        text_only=True,
                    ),
                )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            if message := _assistant_message(
                element,
                vision_details=vision_details,
            ):
                yield message


def _assistant_message(
    element: ModelOutput,
    /,
    *,
    vision_details: Literal["auto", "low", "high"] | Missing,
) -> ChatCompletionAssistantMessageParam | None:
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

    # a message without content nor tool calls is rejected
    if not content and not tool_calls:
        return None

    message: ChatCompletionAssistantMessageParam = ChatCompletionAssistantMessageParam(
        role="assistant",
    )
    # an empty content array is rejected by some servers, the key has to be absent instead
    if content:
        message["content"] = content
    # an empty tool calls array is rejected, the key has to be absent instead
    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


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

    elif output == "auto" or output == "text" or "text" in output:  # noqa: PLR1714
        return omit

    else:
        # silently answering with text would not match the requested modality
        raise NotImplementedError(f"{output} output is not supported by vLLM")
