import json
import random
from collections.abc import AsyncIterable, Generator, Iterable, MutableSequence, Sequence
from typing import Any, TypedDict, cast

from anthropic import Omit, omit
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic.types import (
    CitationsDelta,
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
from anthropic.types.redacted_thinking_block_param import RedactedThinkingBlockParam
from anthropic.types.thinking_block_param import ThinkingBlockParam
from haiway import (
    MISSING,
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
    ModelOutputChunk,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    ModelToolsSelection,
    record_model_invocation,
    record_usage_metrics,
)
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent, TextContent
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
        prefill: Multimodal | None = None,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        async with ctx.scope("model.invocation"):
            config = config or ctx.state(AnthropicConfig)
            record_model_invocation(
                provider=self._provider,
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                stop_sequences=config.stop_sequences,
                thinking_budget=config.thinking_budget,
            )

            messages: Iterable[MessageParam]
            try:
                messages = _context_messages(
                    context,
                    prefill=prefill,
                    output=output,
                )

            except Exception as exc:
                raise ModelInputInvalid(
                    provider=self._provider,
                    model=config.model,
                ) from exc

            tools_list: Iterable[ToolParam] | Omit
            tool_choice: ToolChoiceParam | Omit
            tool_choice, tools_list = _tools_as_tool_params(
                selection=tools.selection,
                specification=tools.specification,
            )

            try:
                tool_accumulator: _ToolAccumulator | None = None
                async with self._client.messages.stream(
                    model=config.model,
                    system=instructions if instructions else omit,
                    messages=messages,
                    temperature=unwrap_missing(
                        config.temperature,
                        default=omit,
                    ),
                    max_tokens=config.max_output_tokens,
                    thinking=_thinking_budget_config(config.thinking_budget),
                    tools=tools_list,
                    tool_choice=tool_choice,
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
                                if tool_accumulator is None:
                                    continue

                                yield ModelToolRequest.of(
                                    tool_accumulator["id"],
                                    tool=tool_accumulator["tool"],
                                    arguments=json.loads("".join(tool_accumulator["arguments"]))
                                    if tool_accumulator["arguments"]
                                    else None,
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

                            case other:
                                raise ModelOutputInvalid(
                                    provider=self._provider,
                                    model=config.model,
                                    reason=f"Unsupported stream event: {other}",
                                )

                assert tool_accumulator is None  # nosec: B101
            except AnthropicRateLimitError as exc:
                if retry_after := exc.response.headers.get("Retry-After"):
                    try:
                        delay = float(retry_after) + random.uniform(0.1, 3.0)  # nosec: B311
                        ctx.record_warning(
                            event="model.rate_limit",
                            attributes={
                                "model.provider": self._provider,
                                "model.name": config.model,
                                "retry_after": delay,
                            },
                        )
                        raise ModelRateLimit(
                            provider=self._provider,
                            model=config.model,
                            retry_after=delay,
                        ) from exc

                    except ValueError:
                        delay = random.uniform(0.3, 5.0)  # nosec: B311
                        ctx.record_warning(
                            event="model.rate_limit",
                            attributes={
                                "model.provider": self._provider,
                                "model.name": config.model,
                                "retry_after": delay,
                            },
                        )
                        raise ModelRateLimit(
                            provider=self._provider,
                            model=config.model,
                            retry_after=delay,
                        ) from exc

                delay = random.uniform(0.3, 5.0)  # nosec: B311
                ctx.record_warning(
                    event="model.rate_limit",
                    attributes={
                        "model.provider": self._provider,
                        "model.name": config.model,
                        "retry_after": delay,
                    },
                )
                raise ModelRateLimit(
                    provider=self._provider,
                    model=config.model,
                    retry_after=delay,
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
    *,
    prefill: Multimodal | None,
    output: ModelOutputSelection,
) -> Generator[MessageParam]:
    for element in context:
        content: list[
            TextBlockParam
            | ImageBlockParam
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
                                Iterable[TextBlockParam | ImageBlockParam],
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

    if prefill is not None:
        yield {
            "role": "assistant",
            "content": _content_elements(
                MultimodalContent.of(prefill),
                cache_type=None,
            ),
        }

    elif output == "json" or isinstance(output, type):
        yield {
            "role": "assistant",
            "content": (
                {
                    "type": "text",
                    "text": "{",
                },
            ),
        }


def _content_elements(
    content: MultimodalContent,
    /,
    cache_type: str | None,
) -> Generator[TextBlockParam | ImageBlockParam]:
    last_cacheable: TextBlockParam | ImageBlockParam | None = None
    for part in content.parts:
        if isinstance(part, TextContent):
            text_block: TextBlockParam = {
                "type": "text",
                "text": part.text,
            }
            last_cacheable = text_block
            yield text_block

        elif isinstance(part, ResourceContent):
            # Only selected image resources are supported by Anthropic message blocks
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


def _thinking_budget_config(
    budget: int | None | Missing,
) -> ThinkingConfigParam | Omit:
    if budget is MISSING:
        return omit

    assert isinstance(budget, int | None)  # nosec: B101

    if budget is None or budget <= 0:
        return {"type": "disabled"}

    return {
        "type": "enabled",
        "budget_tokens": budget,
    }


def _tools_as_tool_params(
    selection: ModelToolsSelection,
    specification: Sequence[ModelToolSpecification],
) -> tuple[ToolChoiceParam | Omit, Iterable[ToolParam] | Omit]:
    if not specification:
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

        case "none":
            return (
                {"type": "none"},
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
