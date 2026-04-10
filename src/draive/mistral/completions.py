import json
import random
from collections.abc import Iterable
from typing import Any, cast
from uuid import uuid4

from haiway import Meta, as_list, ctx, unwrap_missing
from mistralai import (
    ContentChunkTypedDict,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
    ToolTypedDict,
    UsageInfo,
)
from mistralai.models import (
    ChatCompletionRequestToolChoiceTypedDict,
    CompletionEvent,
    CompletionResponseStreamChoice,
    ContentChunk,
    DeltaMessage,
    MessagesTypedDict,
    ResponseFormatTypedDict,
    ToolCall,
)
from mistralai.utils.eventstreaming import EventStreamAsync

from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralChatConfig
from draive.mistral.utils import unwrap_missing_to_none, unwrap_missing_to_unset
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputFailed,
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
    Multimodal,
    MultimodalContent,
    MultimodalContentPart,
    TextContent,
)
from draive.resources import ResourceContent, ResourceReference

__all__ = ("MistralCompletions",)

RATE_LIMIT_STATUS_CODE: int = 429


class MistralCompletions(MistralAPI):
    async def completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: MistralChatConfig | None = None,
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("model.invocation"):
            config = config or ctx.state(MistralChatConfig)
            record_model_invocation(
                provider="mistral",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
            )

            tool_choice: ChatCompletionRequestToolChoiceTypedDict
            tools_list: list[ToolTypedDict]
            tool_choice, tools_list = _tools_as_tool_config(
                tools.specification,
                tool_selection=tools.selection,
            )

            usage: UsageInfo = UsageInfo()
            usage_recorded = False
            try:
                stream: EventStreamAsync[CompletionEvent] = await self._client.chat.stream_async(
                    model=config.model,
                    messages=_build_messages(
                        context=context,
                        instructions=instructions,
                        prefill=prefill,
                    ),
                    temperature=unwrap_missing_to_unset(config.temperature),
                    top_p=unwrap_missing_to_none(config.top_p),
                    max_tokens=unwrap_missing_to_unset(config.max_output_tokens),
                    stop=as_list(unwrap_missing_to_none(config.stop_sequences)),
                    random_seed=unwrap_missing_to_unset(config.seed),
                    response_format=_response_format(output),
                    tools=tools_list,
                    tool_choice=tool_choice,
                )

                tool_accumulator: ToolCall | None = None
                async for event in stream:
                    if event.data.usage is not None:
                        usage = event.data.usage
                        usage_recorded = True

                    if not event.data.choices:
                        continue  # allow usage-only events

                    completion_choice: CompletionResponseStreamChoice = event.data.choices[0]

                    completion_delta: DeltaMessage = completion_choice.delta
                    if content := completion_delta.content:
                        if tool_accumulator is not None:
                            yield ModelToolRequest(
                                identifier=tool_accumulator.id or uuid4().hex,
                                tool=tool_accumulator.function.name,
                                arguments=(
                                    json.loads(tool_accumulator.function.arguments)
                                    if isinstance(tool_accumulator.function.arguments, str)
                                    else tool_accumulator.function.arguments
                                ),
                                meta=Meta.empty,
                            )
                            tool_accumulator = None

                        if isinstance(content, str):
                            yield TextContent.of(content)

                        else:
                            for part in content:
                                yield _content_chunk_as_content_element(part)

                    if tool_calls := completion_delta.tool_calls:
                        for tool_call in tool_calls:
                            if tool_call.index is None:
                                raise ModelOutputFailed(
                                    provider="mistral",
                                    model=config.model,
                                    reason="Invalid completion: missing tool call index",
                                )

                            if tool_accumulator is None:
                                tool_accumulator = tool_call
                                continue  # simply add new entry

                            if tool_accumulator.index != tool_call.index:
                                yield ModelToolRequest(
                                    identifier=tool_accumulator.id or uuid4().hex,
                                    tool=tool_accumulator.function.name,
                                    arguments=(
                                        json.loads(tool_accumulator.function.arguments)
                                        if isinstance(tool_accumulator.function.arguments, str)
                                        else tool_accumulator.function.arguments
                                    ),
                                    meta=Meta.empty,
                                )
                                tool_accumulator = tool_call
                                continue  # replace accumulator entry

                            # "null" is a default value...
                            if tool_call.id and tool_call.id != "null":
                                tool_accumulator.id = tool_call.id

                            if tool_call.function.name:
                                tool_accumulator.function.name += tool_call.function.name

                            tool_arguments: dict[str, Any] | str = tool_call.function.arguments
                            if isinstance(tool_arguments, str):
                                assert isinstance(tool_accumulator.function.arguments, str)  # nosec: B101
                                tool_accumulator.function.arguments += tool_arguments

                            else:
                                assert isinstance(tool_accumulator.function.arguments, dict)  # nosec: B101
                                tool_accumulator.function.arguments = {
                                    # seems there is pyright typing issue around
                                    **cast(dict[str, Any], tool_accumulator.function.arguments),  # pyright: ignore[reportUnnecessaryCast]
                                    **tool_arguments,
                                }

                    if completion_choice.finish_reason == "error":
                        raise ModelOutputFailed(
                            provider="mistral",
                            model=config.model,
                            reason=completion_choice.finish_reason,
                        )

                    if completion_choice.finish_reason in ("length", "model_length"):
                        raise ModelOutputLimit(
                            provider="mistral",
                            model=config.model,
                            max_output_tokens=unwrap_missing(config.max_output_tokens, default=0),
                        )

                if tool_accumulator is not None:
                    yield ModelToolRequest(
                        identifier=tool_accumulator.id or uuid4().hex,
                        tool=tool_accumulator.function.name,
                        arguments=(
                            json.loads(tool_accumulator.function.arguments)
                            if isinstance(tool_accumulator.function.arguments, str)
                            else tool_accumulator.function.arguments
                        ),
                        meta=Meta.empty,
                    )

            except ModelException as exc:
                raise exc

            except Exception as exc:
                if getattr(exc, "status_code", None) == RATE_LIMIT_STATUS_CODE:
                    delay: float = random.uniform(0.3, 3.0)  # nosec: B311
                    ctx.record_warning(
                        event="model.rate_limit",
                        attributes={
                            "model.provider": "mistral",
                            "model.name": config.model,
                            "retry_after": delay,
                        },
                    )
                    raise ModelRateLimit(
                        provider="mistral",
                        model=config.model,
                        retry_after=delay,
                    ) from exc

                raise ModelOutputFailed(
                    provider="mistral",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            finally:
                if usage_recorded:
                    record_usage_metrics(
                        provider="mistral",
                        model=config.model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                    )


def _context_messages(
    context: ModelContext,
) -> Iterable[MessagesTypedDict]:
    for element in context:
        if isinstance(element, ModelInput):
            if user_content := element.content:
                yield {
                    "role": "user",
                    "content": list(_content_chunks(user_content.parts)),
                }

            # Provide tool responses as separate tool messages expected by Mistral
            for tool_response in element.tool_responses:
                yield {
                    "role": "tool",
                    "tool_call_id": tool_response.identifier,
                    "name": tool_response.tool,
                    "content": list(_content_chunks(tool_response.content.parts)),
                }

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            for block in element.output:
                if isinstance(block, MultimodalContent):
                    yield {
                        "role": "assistant",
                        "content": list(_content_chunks(block.parts)),
                    }

                elif isinstance(block, ModelReasoning):
                    continue  # skip reasoning

                else:
                    assert isinstance(block, ModelToolRequest)  # nosec: B101
                    yield {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": block.identifier,
                                "function": {
                                    "name": block.tool,
                                    "arguments": json.dumps(block.arguments),
                                },
                            }
                        ],
                    }


def _build_messages(
    *,
    instructions: ModelInstructions,
    context: ModelContext,
    prefill: Multimodal | None,
) -> list[MessagesTypedDict]:
    messages: list[MessagesTypedDict]
    if instructions:
        messages = [
            {
                "role": "system",
                "content": instructions,
            },
            *_context_messages(context),
        ]

    else:
        messages = list(_context_messages(context))

    if prefill is not None:
        messages.append(
            cast(
                MessagesTypedDict,
                {
                    "role": "assistant",
                    "content": list(_content_chunks(MultimodalContent.of(prefill).parts)),
                },
            )
        )

    return messages


def _content_chunks(
    parts: Iterable[MultimodalContentPart],
) -> Iterable[ContentChunkTypedDict]:
    for element in parts:
        if isinstance(element, TextContent):
            yield {
                "type": "text",
                "text": element.text,
            }

        elif isinstance(element, ResourceReference):
            if not (element.mime_type or "").startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {element.mime_type}")

            yield {
                "type": "image_url",
                "image_url": {
                    "url": element.uri,
                },
            }

        elif isinstance(element, ResourceContent):
            if not (element.mime_type or "").startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {element.mime_type}")

            yield {
                "type": "image_url",
                "image_url": {
                    # ResourceContent.to_data_uri() returns a proper data URI
                    "url": element.to_data_uri(),
                },
            }

        else:
            assert isinstance(element, ArtifactContent)  # nosec: B101
            if element.hidden:
                continue

            yield {
                "type": "text",
                "text": element.to_str(),
            }


def _content_chunk_as_content_element(
    chunk: ContentChunk,
) -> MultimodalContentPart:
    if isinstance(chunk, TextChunk):
        return TextContent.of(chunk.text)

    elif isinstance(chunk, ThinkChunk):
        return ArtifactContent.of(
            ModelReasoning.of(
                *(part.text for part in chunk.thinking if isinstance(part, TextChunk))
            ),
            category="reasoning",
            hidden=True,
        )

    elif isinstance(chunk, ImageURLChunk):
        return ResourceReference.of(
            chunk.image_url if isinstance(chunk.image_url, str) else chunk.image_url.url,
            mime_type="image/png",
        )

    else:
        raise ValueError(f"Unsupported content chunk: {type(chunk)}")


def _tool_specification_as_tool(
    tool: ModelToolSpecification,
) -> ToolTypedDict:
    # Mistral requires a valid JSON schema object for parameters; provide a minimal placeholder
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": cast(dict[str, Any], tool.parameters)
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
) -> tuple[ChatCompletionRequestToolChoiceTypedDict, list[ToolTypedDict]]:
    tools_list: list[ToolTypedDict] = [_tool_specification_as_tool(tool) for tool in (tools or [])]
    if not tools_list:
        return ("none", tools_list)

    if tool_selection == "auto":
        return ("auto", tools_list)

    if tool_selection == "none":
        return ("none", [])

    if tool_selection == "required":
        return ("any", tools_list)

    return (
        {
            "type": "function",
            "function": {
                "name": tool_selection.name,
            },
        },
        tools_list,
    )


def _response_format(
    output: ModelOutputSelection,
) -> ResponseFormatTypedDict | None:
    if output == "json":
        return cast(ResponseFormatTypedDict, {"type": "json_object"})

    if isinstance(output, type):
        # Structured output with DataModel schema

        return cast(
            ResponseFormatTypedDict,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": output.__name__,
                    "schema": output.__SPECIFICATION__,
                },
            },
        )

    return None
