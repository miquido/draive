import json
from collections.abc import Generator, Iterable
from typing import Any, cast

from haiway import Meta, ctx, unwrap_missing
from mistralai.client.models import (
    ChatCompletionStreamRequestMessageTypedDict,
    ChatCompletionStreamRequestToolChoiceTypedDict,
    ChatCompletionStreamRequestToolTypedDict,
    CompletionEvent,
    CompletionResponseStreamChoice,
    ContentChunk,
    ContentChunkTypedDict,
    DeltaMessage,
    ImageURLChunk,
    ResponseFormatTypedDict,
    TextChunk,
    ThinkChunk,
    ThinkChunkTypedDict,
    ToolCall,
    ToolTypedDict,
    UsageInfo,
)
from mistralai.client.utils.eventstreaming import EventStreamAsync

from draive.mistral.api import MistralAPI
from draive.mistral.config import MistralChatConfig
from draive.mistral.utils import (
    unwrap_missing_list_to_unset,
    unwrap_missing_to_unset,
)
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
    ModelReasoningChunk,
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
                top_p=config.top_p,
                seed=config.seed,
                prompt_mode=config.prompt_mode,
            )

            tool_choice: ChatCompletionStreamRequestToolChoiceTypedDict
            tools_list: list[ChatCompletionStreamRequestToolTypedDict]
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
                    ),
                    temperature=unwrap_missing_to_unset(config.temperature),
                    top_p=unwrap_missing_to_unset(config.top_p),
                    max_tokens=unwrap_missing_to_unset(config.max_output_tokens),
                    stop=unwrap_missing_list_to_unset(config.stop_sequences),
                    random_seed=unwrap_missing_to_unset(config.seed),
                    prompt_mode=unwrap_missing_to_unset(config.prompt_mode),
                    response_format=_response_format(output),
                    tools=tools_list,
                    tool_choice=tool_choice,
                )

                tool_accumulator: ToolCall | None = None
                async with stream:
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
                                if request := _tool_request(
                                    tool_accumulator,
                                    model=config.model,
                                ):
                                    yield request

                                tool_accumulator = None

                            if isinstance(content, str):
                                yield TextContent.of(content)

                            else:
                                for part in content:
                                    # reasoning is delivered as a dedicated chunk type
                                    if isinstance(part, ThinkChunk):
                                        # `closed` marks the last fragment of the block,
                                        # keeping its signature paired with its own text
                                        for reasoning in _reasoning_chunks(part):
                                            yield reasoning

                                        continue

                                    element: MultimodalContentPart | None = (
                                        _content_chunk_as_content_element(part)
                                    )
                                    if element is not None:
                                        yield element

                        if tool_calls := completion_delta.tool_calls:
                            for tool_call in tool_calls:
                                if tool_accumulator is None:
                                    tool_accumulator = tool_call
                                    continue  # simply add new entry

                                # `index` defaults to 0 instead of being absent, so a distinct
                                # identifier is what separates parallel calls sharing an index
                                accumulated_identifier: str | None = _tool_call_identifier(
                                    tool_accumulator
                                )
                                call_identifier: str | None = _tool_call_identifier(tool_call)
                                if tool_accumulator.index != tool_call.index or (
                                    call_identifier is not None
                                    and accumulated_identifier is not None
                                    and call_identifier != accumulated_identifier
                                ):
                                    if request := _tool_request(
                                        tool_accumulator,
                                        model=config.model,
                                    ):
                                        yield request

                                    tool_accumulator = tool_call
                                    continue  # replace accumulator entry

                                if call_identifier is not None:
                                    tool_accumulator.id = call_identifier

                                # the name arrives whole within the first fragment,
                                # accumulating it would duplicate a repeated value
                                if not tool_accumulator.function.name and tool_call.function.name:
                                    tool_accumulator.function.name = tool_call.function.name

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
                            # the api reports no detail beyond the reason itself,
                            # naming the request shape helps narrowing it down
                            raise ModelOutputFailed(
                                provider="mistral",
                                model=config.model,
                                reason=(
                                    "Generation terminated with an error"
                                    f" (tools: {len(tools_list)},"
                                    f" tools selection: {
                                        tools.selection
                                        if isinstance(tools.selection, str)
                                        else tools.selection.name
                                    },"
                                    f" output: {output if isinstance(output, str) else 'schema'})"
                                ),
                            )

                        if completion_choice.finish_reason in ("length", "model_length"):
                            raise ModelOutputLimit(
                                provider="mistral",
                                model=config.model,
                                max_output_tokens=unwrap_missing(
                                    config.max_output_tokens, default=0
                                ),
                            )

                if tool_accumulator is not None:
                    if request := _tool_request(
                        tool_accumulator,
                        model=config.model,
                    ):
                        yield request

            except ModelException as exc:
                raise exc

            except Exception as exc:
                if getattr(exc, "status_code", None) == RATE_LIMIT_STATUS_CODE:
                    raise model_rate_limit(
                        provider="mistral",
                        model=config.model,
                        retry_after=None,
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


def _reasoning_chunks(
    chunk: ThinkChunk,
    /,
) -> Generator[ModelReasoningChunk]:
    """Convert a provider thinking chunk into reasoning fragments."""
    meta: Meta = Meta.of(
        # the signature key is omitted rather than carrying None - fragment metadata
        # is merged within a block, so an explicit None would erase a real signature
        {"kind": "thinking", "signature": chunk.signature}
        if isinstance(chunk.signature, str)
        else {"kind": "thinking"}
    )
    texts: list[str] = [
        thinking.text for thinking in chunk.thinking if isinstance(thinking, TextChunk)
    ]
    if not texts:  # a closing chunk carries the signature without any text
        if chunk.closed:
            yield ModelReasoningChunk.of(
                TextContent.empty,
                final=True,
                meta=meta,
            )

        return

    for index, text in enumerate(texts):
        yield ModelReasoningChunk.of(
            TextContent.of(text),
            final=bool(chunk.closed) and index == len(texts) - 1,
            meta=meta,
        )


def _tool_request(
    tool_call: ToolCall,
    /,
    *,
    model: str,
) -> ModelToolRequest | None:
    """Convert an accumulated tool call into a request.

    The identifier is echoed back to the api within the following turn, which
    validates its format - a locally generated one would be rejected there, so a
    call arriving without an identifier is reported and skipped instead.
    """
    identifier: str | None = _tool_call_identifier(tool_call)
    if identifier is None:
        ctx.log_warning(
            f"Mistral reported the {tool_call.function.name} tool call without an"
            " identifier, skipping it - the api would reject a locally generated one"
        )
        return None

    arguments: Any = tool_call.function.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else None

        except Exception as exc:
            raise ModelOutputInvalid(
                provider="mistral",
                model=model,
                reason=f"Tool arguments decoding error - {type(exc).__name__}: {exc}",
            ) from exc

    return ModelToolRequest.of(
        identifier,
        tool=tool_call.function.name,
        arguments=arguments,
    )


def _context_messages(
    context: ModelContext,
) -> Iterable[ChatCompletionStreamRequestMessageTypedDict]:
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
                    # reasoning travels as a dedicated chunk type, dropping it would
                    # break the thinking continuity across turns
                    yield {
                        "role": "assistant",
                        "content": [_reasoning_chunk_param(block)],
                    }

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


def _reasoning_chunk_param(
    block: ModelReasoning,
    /,
) -> ContentChunkTypedDict:
    chunk: ThinkChunkTypedDict = {
        "type": "thinking",
        "thinking": [
            {
                "type": "text",
                "text": block.reasoning.to_str(),
            }
        ],
        "closed": True,
    }
    if signature := block.meta.get_str("signature"):
        chunk["signature"] = signature

    return cast(ContentChunkTypedDict, chunk)


def _build_messages(
    *,
    instructions: ModelInstructions,
    context: ModelContext,
) -> list[ChatCompletionStreamRequestMessageTypedDict]:
    messages: list[ChatCompletionStreamRequestMessageTypedDict]
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

    return messages


def _content_chunks(  # noqa: C901
    parts: Iterable[MultimodalContentPart],
) -> Iterable[ContentChunkTypedDict]:
    for element in parts:
        if isinstance(element, TextContent):
            yield {
                "type": "text",
                "text": element.text,
            }

        elif isinstance(element, ResourceReference):
            if element.mime_type == "application/pdf":
                yield {
                    "type": "document_url",
                    "document_url": element.uri,
                }
                continue

            if not (element.mime_type or "").startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {element.mime_type}")

            yield {
                "type": "image_url",
                "image_url": {
                    "url": element.uri,
                },
            }

        elif isinstance(element, ResourceContent):
            if element.mime_type == "application/pdf":
                yield {
                    "type": "document_url",
                    # a data uri is accepted in place of a remote document location
                    "document_url": element.to_data_uri(),
                }
                continue

            if element.mime_type.startswith("audio"):
                # audio is delivered as bare base64 without the data uri envelope
                yield {
                    "type": "input_audio",
                    "input_audio": element.data,
                }
                continue

            if element.mime_type.startswith("text"):
                # there is no plain text document chunk, inlining keeps the content
                yield {
                    "type": "text",
                    "text": element.to_bytes().decode(),
                }
                continue

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
) -> MultimodalContentPart | None:
    if isinstance(chunk, TextChunk):
        return TextContent.of(chunk.text)

    elif isinstance(chunk, ImageURLChunk):
        return ResourceReference.of(
            chunk.image_url if isinstance(chunk.image_url, str) else chunk.image_url.url,
            mime_type="image/png",
        )

    else:
        # ContentChunk is an open union with a forward compatible fallback variant,
        # skipping unknown chunks keeps the stream alive instead of failing it
        ctx.log_debug(f"Skipping unsupported mistral content chunk: {type(chunk).__name__}")
        return None


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


def _tool_call_identifier(
    tool_call: ToolCall,
    /,
) -> str | None:
    # "null" is the default value of the field
    if tool_call.id and tool_call.id != "null":
        return tool_call.id

    return None


def _tools_as_tool_config(
    tools: Iterable[ModelToolSpecification] | None,
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> tuple[
    ChatCompletionStreamRequestToolChoiceTypedDict,
    list[ChatCompletionStreamRequestToolTypedDict],
]:
    tools_list: list[ChatCompletionStreamRequestToolTypedDict] = [
        _tool_specification_as_tool(tool) for tool in (tools or [])
    ]
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

    if output == "auto" or output == "text" or "text" in output:  # noqa: PLR1714
        return None

    # silently answering with text would not match the requested modality
    raise NotImplementedError(f"{output} output is not supported by Mistral")
