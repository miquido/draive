import json
from collections.abc import (
    Generator,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import Any, Final, Literal, TypedDict, cast

from anthropic import Omit, omit
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.types import (
    CitationsDelta,
    DocumentBlockParam,
    ImageBlockParam,
    InputJSONDelta,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextBlockParam,
    TextDelta,
    ThinkingBlock,
    ThinkingConfigParam,
    ThinkingDelta,
    ToolChoiceParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types.output_config_param import OutputConfigParam
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from haiway import (
    MISSING,
    BasicValue,
    Missing,
    as_dict,
    ctx,
    unwrap_missing,
)

from draive.anthropic.api import AnthropicAPI
from draive.anthropic.config import AnthropicConfig
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputInvalid,
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
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AnthropicMessages",)


class AnthropicMessages(AnthropicAPI):
    async def completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelTools,
        output: ModelOutputSelection,
        config: AnthropicConfig | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        async with ctx.scope("model.invocation"):
            config = config or ctx.state(AnthropicConfig)
            record_model_invocation(
                provider=self._provider,
                model=config.model,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
                thinking=config.thinking,
                effort=config.effort,
            )

            messages: list[MessageParam]
            try:
                # eagerly materialize to convert context errors to ModelInputInvalid here
                messages = list(_context_messages(context))

            except Exception as exc:
                raise ModelInputInvalid(
                    provider=self._provider,
                    model=config.model,
                    reason=str(exc),
                ) from exc

            tools_list: Iterable[ToolParam] | Omit
            tool_choice: ToolChoiceParam | Omit
            tool_choice, tools_list = _tools_as_tool_params(
                selection=tools.selection,
                specification=tools.specification,
            )

            try:
                tool_accumulator: _ToolAccumulator | None = None
                # kind of the reasoning block currently open, it has to be closed on
                # its `content_block_stop` to keep its signature from merging into the
                # next block within the same message
                reasoning_block: str | None = None
                async with self._client.messages.stream(
                    model=config.model,
                    system=instructions if instructions else omit,
                    messages=messages,
                    max_tokens=config.max_output_tokens,
                    thinking=_thinking_config(config.thinking),
                    tools=tools_list,
                    tool_choice=tool_choice,
                    output_config=_output_config(
                        output,
                        effort=config.effort,
                    ),
                    stop_sequences=unwrap_missing(
                        cast(Any, config.stop_sequences),
                        default=omit,
                    ),
                ) as stream:
                    async for event in stream:
                        match event.type:
                            case "content_block_delta":
                                assert isinstance(event, RawContentBlockDeltaEvent)  # nosec: B101
                                match event.delta.type:
                                    case "text_delta":
                                        assert isinstance(event.delta, TextDelta)  # nosec: B101
                                        yield TextContent.of(event.delta.text)

                                    case "thinking_delta":
                                        assert isinstance(event.delta, ThinkingDelta)  # nosec: B101
                                        yield ModelReasoningChunk.of(
                                            TextContent.of(event.delta.thinking),
                                            meta={"kind": "thinking"},
                                        )

                                    case "input_json_delta":
                                        assert isinstance(event.delta, InputJSONDelta)  # nosec: B101
                                        assert tool_accumulator is not None  # nosec: B101
                                        tool_accumulator["arguments"].append(
                                            event.delta.partial_json
                                        )

                                    case "signature_delta":
                                        assert isinstance(event.delta, SignatureDelta)  # nosec: B101
                                        yield ModelReasoningChunk.of(
                                            TextContent.empty,
                                            meta={
                                                "kind": "thinking",
                                                "signature": event.delta.signature,
                                            },
                                        )

                                    case "citations_delta":
                                        assert isinstance(event.delta, CitationsDelta)  # nosec: B101
                                        pass  # unsupported

                            case "content_block_start":
                                assert isinstance(event, RawContentBlockStartEvent)  # nosec: B101
                                match event.content_block.type:
                                    case "thinking":
                                        assert isinstance(event.content_block, ThinkingBlock)  # nosec: B101
                                        reasoning_block = "thinking"
                                        if event.content_block.thinking:
                                            yield ModelReasoningChunk.of(
                                                TextContent.of(event.content_block.thinking),
                                                meta={
                                                    "kind": "thinking",
                                                    "signature": event.content_block.signature,
                                                },
                                            )

                                    case "text":
                                        continue  # actual content arrives in text_delta events

                                    case "tool_use":
                                        assert isinstance(event.content_block, ToolUseBlock)  # nosec: B101
                                        assert not event.content_block.input  # nosec: B101
                                        tool_accumulator = {
                                            "id": event.content_block.id,
                                            "tool": event.content_block.name,
                                            "arguments": [],
                                        }

                                    case "redacted_thinking":
                                        assert isinstance(
                                            event.content_block, RedactedThinkingBlock
                                        )  # nosec: B101
                                        reasoning_block = "redacted_thinking"
                                        yield ModelReasoningChunk.of(
                                            TextContent.empty,
                                            meta={
                                                "kind": "redacted_thinking",
                                                "data": event.content_block.data,
                                            },
                                        )

                                    case other:
                                        raise ModelOutputInvalid(
                                            provider=self._provider,
                                            model=config.model,
                                            reason=f"Unsupported content block: {other}",
                                        )

                            case "content_block_stop":
                                if reasoning_block is not None:
                                    # closes the block, keeping its signature paired
                                    # with the thinking text it was produced for
                                    yield ModelReasoningChunk.of(
                                        TextContent.empty,
                                        final=True,
                                        meta={"kind": reasoning_block},
                                    )
                                    reasoning_block = None
                                    continue

                                if tool_accumulator is None:
                                    continue

                                # a tool without arguments still delivers a single, empty
                                # json fragment - decoding it would fail on empty input
                                accumulated_arguments: str = "".join(
                                    tool_accumulator["arguments"]
                                ).strip()
                                decoded_arguments: Any
                                try:
                                    decoded_arguments = (
                                        json.loads(accumulated_arguments)
                                        if accumulated_arguments
                                        else None
                                    )

                                except Exception as exc:
                                    raise ModelOutputInvalid(
                                        provider=self._provider,
                                        model=config.model,
                                        reason=(
                                            "Tool arguments decoding error - "
                                            f"{type(exc).__name__}: {exc}"
                                        ),
                                    ) from exc

                                # a tool call takes named arguments, anything else
                                # would reach the tool as an unusable payload
                                if decoded_arguments is not None and not isinstance(
                                    decoded_arguments, Mapping
                                ):
                                    raise ModelOutputInvalid(
                                        provider=self._provider,
                                        model=config.model,
                                        reason=(
                                            "Tool arguments are not an object -"
                                            f" {type(decoded_arguments).__name__}"
                                        ),
                                    )

                                yield ModelToolRequest.of(
                                    tool_accumulator["id"],
                                    tool=tool_accumulator["tool"],
                                    arguments=cast(
                                        Mapping[str, BasicValue] | None,
                                        decoded_arguments,
                                    ),
                                )
                                tool_accumulator = None

                            case "message_delta":
                                assert isinstance(event, RawMessageDeltaEvent)  # nosec: B101
                                record_usage_metrics(
                                    provider=self._provider,
                                    model=config.model,
                                    input_tokens=event.usage.input_tokens,
                                    cached_input_tokens=event.usage.cache_read_input_tokens,
                                    output_tokens=event.usage.output_tokens,
                                )

                                match event.delta.stop_reason:
                                    case "end_turn" | "tool_use" | "pause_turn":
                                        continue  # let it finish

                                    case "max_tokens":
                                        raise ModelOutputLimit(
                                            provider=self._provider,
                                            model=config.model,
                                            max_output_tokens=unwrap_missing(
                                                config.max_output_tokens, default=0
                                            ),
                                        )

                                    case "model_context_window_exceeded":
                                        # an exceeded context window is an input side limit -
                                        # reporting it as an output limit would suggest a
                                        # truncated result that can be continued
                                        raise ModelOutputFailed(
                                            provider=self._provider,
                                            model=config.model,
                                            reason="model_context_window_exceeded",
                                        )

                                    case "refusal":
                                        raise ModelOutputFailed(
                                            provider=self._provider,
                                            model=config.model,
                                            reason="refusal",
                                        )

                                    case "stop_sequence":
                                        continue  # let it finish

                                    case other:
                                        raise ModelOutputFailed(
                                            provider=self._provider,
                                            model=config.model,
                                            reason=f"Unsupported stop reason: {other}",
                                        )

                            case "message_start":
                                assert isinstance(event, RawMessageStartEvent)  # nosec: B101
                                if event.message.content:
                                    pass  # TODO: FIXME: provide initial data

                            case "message_stop":
                                continue  # let it finish

                            case "text" | "input_json" | "thinking" | "signature" | "citation":
                                # Cumulative snapshot events emitted by the SDK stream helper;
                                # the underlying data already arrives via content_block_delta.
                                continue

                assert tool_accumulator is None  # nosec: B101
            except AnthropicRateLimitError as exc:
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


class _ToolAccumulator(TypedDict):
    id: str
    tool: str
    arguments: MutableSequence[str]


def _context_messages(  # noqa: C901, PLR0912
    context: ModelContext,
    /,
) -> Generator[MessageParam]:
    if context and isinstance(context[-1], ModelOutput):
        # a trailing model turn would be sent as an assistant prefill, which every
        # model available through `AnthropicConfig` rejects with a request error
        raise ValueError(
            "Anthropic context has to end with a model input,"
            " a trailing model output would become an unsupported assistant prefill"
        )

    for element in context:
        content: list[
            TextBlockParam
            | ImageBlockParam
            | DocumentBlockParam
            | ThinkingBlockParam
            | RedactedThinkingBlockParam
            | ToolUseBlockParam
            | ToolResultBlockParam
        ] = []

        if isinstance(element, ModelInput):
            for block in element.input:
                if isinstance(block, MultimodalContent):
                    content.extend(
                        _content_elements(
                            block,
                            cache_type=element.meta.get_str("cache"),
                        )
                    )

                else:
                    content.append(
                        {
                            "tool_use_id": block.identifier,
                            "type": "tool_result",
                            "is_error": block.status == "error",
                            "content": cast(  # there will be no thinking within tool results
                                Iterable[TextBlockParam | ImageBlockParam],  # nor documents
                                _content_elements(
                                    block.content,
                                    cache_type=None,
                                ),
                            ),
                        }
                    )

            yield {
                "role": "user",
                "content": content,
            }

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            for block in element.output:
                if isinstance(block, MultimodalContent):
                    content.extend(
                        _content_elements(
                            block,
                            cache_type=element.meta.get_str("cache"),
                        )
                    )

                elif isinstance(block, ModelReasoning):
                    match block.meta.kind:
                        case "thinking":
                            content.append(
                                {
                                    "type": "thinking",
                                    "thinking": block.reasoning.to_str(),
                                    "signature": block.meta.get_str(
                                        "signature",
                                        default="",
                                    ),
                                }
                            )

                        case "redacted_thinking":
                            content.append(
                                {
                                    "type": "redacted_thinking",
                                    "data": block.meta.get_str("data", default=""),
                                }
                            )

                        case other:
                            raise ValueError(f"Unsupported reasoning element: {other}")

                else:
                    assert isinstance(block, ModelToolRequest)  # nosec: B101
                    content.append(
                        {
                            "id": block.identifier,
                            "type": "tool_use",
                            "name": block.tool,
                            "input": as_dict(block.arguments),
                        }
                    )

            yield {
                "role": "assistant",
                "content": content,
            }


# documents are delivered through a dedicated block, everything else has to be an image
_DOCUMENT_MIME_TYPES: Final[frozenset[str]] = frozenset(("application/pdf", "text/plain"))


def _content_elements(  # noqa: C901
    content: MultimodalContent,
    /,
    cache_type: str | None,
) -> Generator[TextBlockParam | ImageBlockParam | DocumentBlockParam]:
    last_cacheable: TextBlockParam | ImageBlockParam | DocumentBlockParam | None = None
    for part in content.parts:
        if isinstance(part, TextContent):
            text_block: TextBlockParam = {
                "type": "text",
                "text": part.text,
            }
            last_cacheable = text_block
            yield text_block

        elif isinstance(part, ResourceContent):
            if part.mime_type in _DOCUMENT_MIME_TYPES:
                document_block: DocumentBlockParam = {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": part.data,
                    }
                    if part.mime_type == "application/pdf"
                    else {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": part.to_bytes().decode(),
                    },
                }
                last_cacheable = document_block
                yield document_block
                continue

            # anything else has to be one of the supported image formats
            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": cast(Any, part.mime_type),
                    "data": part.data,
                },
            }
            last_cacheable = image_block
            yield image_block

        elif isinstance(part, ResourceReference):
            if part.mime_type == "application/pdf":
                document_block: DocumentBlockParam = {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": part.uri,
                    },
                }
                last_cacheable = document_block
                yield document_block
                continue

            # Only image resources are supported by Anthropic message blocks
            if part.mime_type and not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported message content mime type: {part.mime_type}")

            # TODO: auto resolve non http resources
            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": part.uri,
                },
            }

            last_cacheable = image_block
            yield image_block

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            # Skip artifacts that are marked as hidden
            if part.hidden:
                continue

            text_block: TextBlockParam = {
                "type": "text",
                "text": part.to_str(),
            }
            last_cacheable = text_block
            yield text_block

    if cache_type is None or last_cacheable is None:
        return

    # insert cache marker to the last cacheable element
    last_cacheable["cache_control"] = {  # pyright: ignore[reportGeneralTypeIssues]
        "type": cache_type,
    }


def _output_config(
    output: ModelOutputSelection,
    /,
    *,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | Missing,
) -> OutputConfigParam | Omit:
    config: OutputConfigParam = {}
    # only a schema backed selection maps onto the structured output API - a
    # schema-less json request has no dedicated API mode, so it stays as
    # reliable as the caller's own instructions make it
    if isinstance(output, type):
        config["format"] = {
            "type": "json_schema",
            "schema": as_dict(output.__SPECIFICATION__),
        }

    elif output not in ("auto", "text", "json") and "text" not in output:
        # silently answering with text would not match the requested modality
        raise NotImplementedError(f"{output} output is not supported by Anthropic")

    if effort is not MISSING:
        config["effort"] = cast(Literal["low", "medium", "high", "xhigh", "max"], effort)

    return config or omit


def _thinking_config(
    thinking: Literal["adaptive", "disabled"] | Missing,
    /,
) -> ThinkingConfigParam | Omit:
    # thinking is adaptive unless requested otherwise, omitting it keeps that default
    # instead of sending a mode which some models refuse
    if thinking is MISSING:
        return omit

    if thinking == "disabled":
        return {"type": "disabled"}

    return {"type": "adaptive"}


def _tools_as_tool_params(
    selection: ModelToolsSelection,
    specification: Sequence[ModelToolSpecification],
) -> tuple[ToolChoiceParam | Omit, Iterable[ToolParam] | Omit]:
    # declaring the tools together with a "none" choice makes the model answer with
    # no content at all - omitting them keeps the same semantics while preserving the
    # response. Tool blocks already present within the context remain valid without
    # their declarations.
    if not specification or selection == "none":
        return (omit, omit)

    tool_params: list[ToolParam] = []
    for tool in specification:
        input_schema: dict[str, Any]
        if parameters := tool.parameters:
            input_schema = cast(dict[str, Any], parameters)

        else:
            # Anthropic requires input_schema; provide an empty object schema when None
            input_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }

        tool_params.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": input_schema,
            }
        )

    match selection:
        case "auto":
            return (
                {"type": "auto"},
                tool_params,
            )

        case "required":
            return (
                {"type": "any"},
                tool_params,
            )

        case specific_tool:
            return (
                {
                    "type": "tool",
                    "name": specific_tool.name,
                },
                tool_params,
            )
