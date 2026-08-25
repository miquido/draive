import json
from base64 import b64decode, b64encode
from collections.abc import (
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import Any, Final, Literal, TypedDict, cast

from haiway import MISSING, BasicValue, Missing, asynchronous, ctx, unwrap_missing

from draive.bedrock.api import BedrockAPI
from draive.bedrock.config import BedrockChatConfig
from draive.bedrock.models import (
    ChatMessage,
    ChatMessageContent,
    ChatMessageImage,
    ChatMessageText,
    ChatTool,
)
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputInvalid,
    ModelInstructions,
    ModelOutput,
    ModelOutputChunk,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelOutputStream,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
)
from draive.models.metrics import (
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

__all__ = ("BedrockConverse",)


# image formats supported by the Converse API
_IMAGE_FORMATS: Final[Mapping[str, Literal["png", "jpeg", "gif", "webp"]]] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}
_IMAGE_MIME_TYPES: Final[Mapping[str, str]] = {
    image_format: mime_type for mime_type, image_format in _IMAGE_FORMATS.items()
}


class BedrockConverse(BedrockAPI):
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: BedrockChatConfig | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        return self._completion_stream(
            instructions=instructions,
            context=context,
            tools=tools,
            output=output,
            config=config or ctx.state(BedrockChatConfig),
            **extra,
        )

    async def _completion_stream(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelOutputSelection,
        config: BedrockChatConfig,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("model.invocation"):
            record_model_invocation(
                provider="bedrock",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
                top_p=config.top_p,
            )

            messages: list[ChatMessage]
            try:
                messages = _request_messages(
                    context,
                    output=output,
                )

            except Exception as exc:
                raise ModelInputInvalid(
                    provider="bedrock",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            # the sdk delivers an event stream object, iterated through its own iterator
            stream: Any
            try:
                stream = await self._converse_stream(
                    _request_parameters(
                        instructions=instructions,
                        messages=messages,
                        tools=tools,
                        config=config,
                    )
                )

            except Exception as exc:
                raise _request_failure(
                    exc,
                    model=config.model,
                ) from exc

            # a tool call arrives as a start event, argument deltas and a stop event
            tool_accumulator: _ToolAccumulator | None = None
            events: Iterator[Mapping[str, Any]] = iter(stream)
            try:
                while (event := await self._next_stream_event(events)) is not None:
                    match event:
                        case {"contentBlockStart": {"start": {"toolUse": tool_use}}}:
                            tool_accumulator = {
                                "id": tool_use["toolUseId"],
                                "tool": tool_use["name"],
                                "arguments": [],
                            }

                        case {"contentBlockDelta": {"delta": delta}}:
                            for chunk in _delta_chunks(
                                delta,
                                tool_accumulator=tool_accumulator,
                                model=config.model,
                            ):
                                yield chunk

                        case {"contentBlockStop": _}:
                            if tool_accumulator is None:
                                continue  # a content block needs no completion

                            yield _accumulated_tool_request(
                                tool_accumulator,
                                model=config.model,
                            )
                            tool_accumulator = None

                        case {"metadata": {"usage": usage}}:
                            record_usage_metrics(
                                provider="bedrock",
                                model=config.model,
                                input_tokens=usage.get("inputTokens"),
                                cached_input_tokens=usage.get("cacheReadInputTokens"),
                                output_tokens=usage.get("outputTokens"),
                            )

                        case {"messageStop": {"stopReason": str() as stop_reason}}:
                            _verify_stop_reason(
                                stop_reason,
                                model=config.model,
                                max_output_tokens=config.max_output_tokens,
                            )

                        case _:
                            # the remaining members are modeled failures
                            if failure := _stream_failure(
                                event,
                                model=config.model,
                            ):
                                raise failure

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="bedrock",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            finally:
                # release the http stream, iteration may have ended
                # before the response was completed
                await self._close_stream(stream)

    @asynchronous
    def _converse_stream(
        self,
        parameters: dict[str, Any],
    ) -> Any:
        return self._client.converse_stream(**parameters)["stream"]

    @asynchronous
    def _next_stream_event(
        self,
        events: Iterator[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        # the sdk stream is a blocking iterator, stepping it off the event loop
        # keeps the loop free while waiting for the next event
        return next(events, None)

    @asynchronous
    def _close_stream(
        self,
        stream: Any,
    ) -> None:
        # closing the event stream releases the underlying blocking response body
        stream.close()


class _ToolAccumulator(TypedDict):
    id: str
    tool: str
    arguments: MutableSequence[str]


def _request_messages(
    context: ModelContext,
    /,
    *,
    output: ModelOutputSelection,
) -> list[ChatMessage]:
    # the Converse API offers neither a schema backed nor a schema-less json
    # mode, so every json selection falls all the way back to plain output -
    # the requested selection is only verified here
    _verify_output(output)

    return _context_messages(context)


def _delta_chunks(
    delta: Mapping[str, Any],
    /,
    *,
    tool_accumulator: _ToolAccumulator | None,
    model: str,
) -> Generator[ModelOutputChunk]:
    match delta:
        case {"text": str() as text}:
            yield TextContent.of(text)

        case {"toolUse": {"input": str() as arguments}}:
            if tool_accumulator is None:
                raise ModelOutputInvalid(
                    provider="bedrock",
                    model=model,
                    reason="Tool arguments delivered without a tool call",
                )

            tool_accumulator["arguments"].append(arguments)

        case {"reasoningContent": reasoning}:
            match reasoning:
                case {"signature": str() as signature}:
                    # the signature closes the block it was produced for
                    yield ModelReasoningChunk.of(
                        TextContent.empty,
                        final=True,
                        meta={
                            "kind": "reasoning",
                            "signature": signature,
                        },
                    )

                case {"redactedContent": bytes() as redacted}:
                    yield ModelReasoningChunk.of(
                        TextContent.empty,
                        final=True,
                        meta={
                            "kind": "redacted_reasoning",
                            "data": b64encode(redacted).decode(),
                        },
                    )

                case {"text": str() as reasoning_text}:
                    yield ModelReasoningChunk.of(
                        TextContent.of(reasoning_text),
                        meta={"kind": "reasoning"},
                    )

                case _:
                    pass  # nothing to report

        case {"image": {"source": {"bytes": bytes() as data}, "format": str() as data_format}}:
            media_type: str | None = _IMAGE_MIME_TYPES.get(data_format)
            if media_type is None:
                raise ModelOutputInvalid(
                    provider="bedrock",
                    model=model,
                    reason=f"Unsupported output image format: {data_format}",
                )

            yield ResourceContent.of(
                data,
                mime_type=media_type,
            )

        case _:
            pass  # citations and tool results are not reported


def _accumulated_tool_request(
    accumulator: _ToolAccumulator,
    /,
    *,
    model: str,
) -> ModelToolRequest:
    # a tool without arguments delivers no fragment at all
    accumulated: str = "".join(accumulator["arguments"]).strip()
    arguments: Any
    try:
        arguments = json.loads(accumulated) if accumulated else None

    except Exception as exc:
        raise ModelOutputInvalid(
            provider="bedrock",
            model=model,
            reason=f"Tool arguments decoding error - {type(exc).__name__}: {exc}",
        ) from exc

    if arguments is not None and not isinstance(arguments, Mapping):
        raise ModelOutputInvalid(
            provider="bedrock",
            model=model,
            reason=f"Tool arguments are not an object - {type(arguments).__name__}",
        )

    return ModelToolRequest.of(
        accumulator["id"],
        tool=accumulator["tool"],
        arguments=cast(Mapping[str, BasicValue] | None, arguments),
    )


# stop reasons describing a terminal failure instead of a regular completion
_FAILURE_STOP_REASONS: Final[frozenset[str]] = frozenset(
    (
        "guardrail_intervened",
        "content_filtered",
        "malformed_model_output",
        "malformed_tool_use",
        # an exceeded context window is an input side limit - reporting it as an
        # output limit would suggest a truncated result that can be continued
        "model_context_window_exceeded",
    )
)


def _verify_stop_reason(
    stop_reason: str,
    /,
    *,
    model: str,
    max_output_tokens: int | Missing,
) -> None:
    if stop_reason == "max_tokens":
        raise ModelOutputLimit(
            provider="bedrock",
            model=model,
            max_output_tokens=unwrap_missing(max_output_tokens, default=0),
        )

    if stop_reason in _FAILURE_STOP_REASONS:
        raise ModelOutputFailed(
            provider="bedrock",
            model=model,
            reason=stop_reason,
        )


# failures delivered as stream members instead of raised by the request
_STREAM_FAILURE_MEMBERS: Final[Mapping[str, str]] = {
    "throttlingException": "ThrottlingException",
    "serviceUnavailableException": "ServiceUnavailableException",
    "validationException": "ValidationException",
    "modelStreamErrorException": "ModelStreamErrorException",
    "internalServerException": "InternalServerException",
}


def _stream_failure(
    event: Mapping[str, Any],
    /,
    *,
    model: str,
) -> ModelException | None:
    for member, code in _STREAM_FAILURE_MEMBERS.items():
        if member not in event:
            continue

        if code == "ThrottlingException":
            return model_rate_limit(
                provider="bedrock",
                model=model,
                retry_after=None,
            )

        if code == "ValidationException":
            return ModelInputInvalid(
                provider="bedrock",
                model=model,
            )

        return ModelOutputFailed(
            provider="bedrock",
            model=model,
            reason=f"{code}: {event[member].get('message', '')}",
        )

    return None


def _request_parameters(
    *,
    instructions: ModelInstructions,
    messages: list[ChatMessage],
    tools: ModelTools,
    config: BedrockChatConfig,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "modelId": config.model,
        "messages": messages,
        "inferenceConfig": {
            "temperature": config.temperature,
        },
    }
    if instructions:
        parameters["system"] = [{"text": instructions}]

    if tool_config := _tools_as_tool_config(
        tools.specification,
        tool_selection=tools.selection,
    ):
        parameters["toolConfig"] = tool_config

    if config.max_output_tokens is not MISSING:
        parameters["inferenceConfig"]["maxTokens"] = config.max_output_tokens

    if config.top_p is not MISSING:
        parameters["inferenceConfig"]["topP"] = config.top_p

    if config.stop_sequences:
        parameters["inferenceConfig"]["stopSequences"] = config.stop_sequences

    if config.guardrail_identifier is not MISSING and config.guardrail_version is not MISSING:
        parameters["guardrailConfig"] = {
            "guardrailIdentifier": config.guardrail_identifier,
            "guardrailVersion": config.guardrail_version,
            # the guardrail has to see the whole response to intervene on it
            "streamProcessingMode": "sync",
        }

    return parameters


# request failures are reported through modeled error codes instead of dedicated types
_THROTTLING_ERROR_CODES: Final[frozenset[str]] = frozenset(
    (
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceQuotaExceededException",
    )
)
_INPUT_ERROR_CODES: Final[frozenset[str]] = frozenset(
    (
        "ValidationException",
        "ServiceUnavailableException",
    )
)


def _request_failure(
    exception: Exception,
    /,
    *,
    model: str,
) -> ModelException:
    # boto does not expose its modeled exceptions without the client, the error code
    # within the response is what identifies the failure
    code: str = ""
    match getattr(exception, "response", None):
        case {"Error": {"Code": str() as error_code}}:
            code = error_code

        case _:
            code = type(exception).__name__

    if code in _THROTTLING_ERROR_CODES:
        return model_rate_limit(
            provider="bedrock",
            model=model,
            retry_after=None,
        )

    if code in _INPUT_ERROR_CODES:
        return ModelInputInvalid(
            provider="bedrock",
            model=model,
        )

    return ModelOutputFailed(
        provider="bedrock",
        model=model,
        reason=str(exception),
    )


def _verify_output(
    output: ModelOutputSelection,
    /,
) -> None:
    """Reject output selections the Converse API cannot deliver.

    Both json selections - schema backed and schema-less - are accepted and
    fall back to plain output, since the API implements neither mode. Shaping
    the answer is left to the caller's instructions, which
    ``ModelGeneration.generate`` can extend with the schema on request.

    Parameters
    ----------
    output : ModelOutputSelection
        Requested output modality selection.

    Raises
    ------
    NotImplementedError
        Raised for modalities the provider cannot produce.
    """
    if isinstance(output, type) or output == "json":
        return  # no dedicated mode, falls back to plain output

    if output in ("auto", "text") or "text" in output:
        return

    # silently answering with text would not match the requested modality
    raise NotImplementedError(f"{output} output is not supported by Bedrock")


def _context_messages(  # noqa: C901, PLR0912
    context: ModelContext,
) -> list[ChatMessage]:
    role: Literal["user", "assistant"] = "user"
    content: list[ChatMessageContent] = []
    messages: list[ChatMessage] = []

    def flush(
        current_role: Literal["user", "assistant"],
    ) -> ChatMessage | None:
        nonlocal content
        if not content:
            return None

        message: ChatMessage = {
            "role": current_role,
            "content": content,
        }
        content = []
        return message

    for element in context:
        if isinstance(element, ModelInput):
            if role != "user":
                if message := flush(role):
                    messages.append(message)

                role = "user"

            for block in element.input:
                if isinstance(block, MultimodalContent):
                    content.extend(_convert_content(block.parts))

                else:
                    assert isinstance(block, ModelToolResponse)  # nosec: B101
                    # tool response -> toolResult
                    content.append(
                        {
                            "toolResult": {
                                "toolUseId": block.identifier,
                                "content": cast(
                                    list[ChatMessageText | ChatMessageImage],
                                    _convert_content(block.content.parts),
                                ),
                                "status": "error" if block.status == "error" else "success",
                            }
                        }
                    )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            if role != "assistant":
                if message := flush(role):
                    messages.append(message)

                role = "assistant"

            for block in element.output:
                if isinstance(block, MultimodalContent):
                    content.extend(_convert_content(block.parts))

                elif isinstance(block, ModelReasoning):
                    continue  # skip reasoning

                else:
                    assert isinstance(block, ModelToolRequest)  # nosec: B101
                    # tool request -> toolUse
                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": block.identifier,
                                "name": block.tool,
                                "input": _json_document(block.arguments),
                            }
                        }
                    )

    if message := flush(role):
        messages.append(message)

    return messages


def _convert_content(
    parts: Sequence[MultimodalContentPart],
) -> list[ChatMessageContent]:
    converted: list[ChatMessageContent] = []
    for part in parts:
        if isinstance(part, TextContent):
            converted.append({"text": part.text})

        elif isinstance(part, ResourceContent):
            # Only selected image resources are supported by Bedrock messages
            fmt: Literal["png", "jpeg", "gif", "webp"] | None = _IMAGE_FORMATS.get(part.mime_type)
            if fmt is None:
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            converted.append(
                {
                    "image": {
                        "format": fmt,
                        "source": {
                            # ResourceContent.data is base64-encoded,
                            # Bedrock expects raw bytes
                            "bytes": b64decode(part.data),
                        },
                    }
                }
            )

        elif isinstance(part, ResourceReference):
            # Bedrock image message blocks only accept raw bytes. We can fetch
            # the resource content if a repository is configured; otherwise this
            # cannot be sent as-is. For now, raise a clear error.
            raise ValueError(
                "ResourceReference in message content is not supported for Bedrock. "
                "Provide inline ResourceContent or let us implement fetching via "
                "ResourcesRepository."
            )

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            # Skip artifacts that are marked as hidden
            if part.hidden:
                continue

            converted.append({"text": part.to_str()})

    return converted


def _json_document(value: Any) -> Any:
    # document parameters are validated to contain only plain json values
    match value:
        case str() | bool() | int() | float() | None:
            return value

        case Mapping():
            mapping: Mapping[str, Any] = cast(Mapping[str, Any], value)
            return {key: _json_document(element) for key, element in mapping.items()}

        case bytes() | bytearray():
            return value  # not a valid document value, request validation reports it

        case Sequence():
            sequence: Sequence[Any] = cast(Sequence[Any], value)
            return [_json_document(element) for element in sequence]

        case other:
            return other


def _convert_tool(tool: ModelToolSpecification) -> ChatTool:
    input_schema: Any
    if parameters := tool.parameters:
        input_schema = _json_document(parameters)

    else:
        # Bedrock requires inputSchema document, provide an empty object schema when None
        input_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    converted: ChatTool = {
        "name": tool.name,
        "inputSchema": {"json": input_schema},
    }

    # Bedrock rejects an empty description, include it only when available
    if description := tool.description:
        converted["description"] = description

    return converted


def _tools_as_tool_config(
    tools: Iterable[ModelToolSpecification] | None,
    /,
    *,
    tool_selection: ModelToolsSelection,
) -> dict[str, Any] | None:
    toolChoice: dict[str, Any]
    if tool_selection == "auto":
        toolChoice = {"auto": {}}

    elif tool_selection == "required":
        toolChoice = {"any": {}}

    elif tool_selection == "none":
        return None

    else:
        toolChoice = {
            "tool": {
                "name": tool_selection.name,
            },
        }

    tools_list = [{"toolSpec": _convert_tool(tool)} for tool in tools or ()]
    if not tools_list:
        return None

    return {"tools": tools_list, "toolChoice": toolChoice}
