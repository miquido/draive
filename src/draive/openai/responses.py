import json
import random
from collections.abc import Generator, Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from haiway import MISSING, Meta, Missing, as_dict, ctx, unwrap_missing
from openai import Omit, omit
from openai import RateLimitError as OpenAIRateLimitError
from openai.types import ReasoningEffort
from openai.types.responses import (
    ResponseAudioDeltaEvent,
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseInputImageContentParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextContentParam,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessageParam,
    ResponseOutputTextParam,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseTextConfigParam,
    ResponseTextDeltaEvent,
    ToolChoiceFunctionParam,
    ToolChoiceOptions,
    ToolParam,
)
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput, Message
from openai.types.responses.response_input_param import (
    ImageGenerationCall as ImageGenerationCallParam,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_output_text_param import (
    Annotation,
    AnnotationContainerFileCitation,
    AnnotationFileCitation,
    AnnotationFilePath,
    AnnotationURLCitation,
)
from openai.types.shared_params.reasoning import Reasoning

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
    ModelRateLimit,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    record_model_invocation,
    record_usage_metrics,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIResponsesConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIResponses",)


class OpenAIResponses(OpenAIAPI):
    async def completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelTools,
        output: ModelOutputSelection,
        config: OpenAIResponsesConfig | None = None,
        cache_key: str | None = None,
        **extra: Any,
    ) -> ModelOutputStream:
        assert isinstance(config, OpenAIResponsesConfig | None)  # nosec: B101
        async with ctx.scope("model.invocation"):
            config = config or ctx.state(OpenAIResponsesConfig)
            record_model_invocation(
                provider="openai",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                verbosity=config.verbosity,
                reasoning=config.reasoning,
                service_tier=config.service_tier,
            )

            input_context: list[ResponseInputItemParam]
            try:
                input_context = list(
                    _context_to_params(
                        context,
                        vision_details=config.vision_details,
                    )
                )

            except Exception as exc:
                raise ModelInputInvalid(
                    provider="openai",
                    model=config.model,
                ) from exc

            try:
                async with self._client.responses.stream(
                    model=config.model,
                    instructions=instructions or omit,
                    input=input_context,
                    temperature=unwrap_missing(
                        config.temperature,
                        default=omit,
                    ),
                    tool_choice=_tool_choice(tools),
                    tools=_tools_as_tool_params(tools.specification),
                    parallel_tool_calls=True if tools else omit,
                    text=_text_output(
                        output,
                        verbosity=config.verbosity,
                    ),
                    reasoning=(
                        Reasoning(
                            effort=cast(ReasoningEffort, config.reasoning),
                            summary=config.reasoning_summary,
                        )
                        if isinstance(config.reasoning, str)
                        else omit
                    ),
                    max_output_tokens=unwrap_missing(
                        config.max_output_tokens,
                        default=omit,
                    ),
                    service_tier=config.service_tier,
                    truncation=config.truncation,
                    safety_identifier=unwrap_missing(
                        config.safety_identifier,
                        default=omit,
                    ),
                    prompt_cache_key=cache_key or omit,
                    include=["reasoning.encrypted_content"]
                    # for gpt-5 model family we need to request encrypted reasoning
                    if "gpt-5" in config.model.lower()
                    else omit,
                    store=False,
                ) as stream:
                    async for event in stream:
                        match event.type:
                            case "response.output_text.delta":
                                assert isinstance(event, ResponseTextDeltaEvent)  # nosec: B101
                                yield TextContent(text=event.delta)

                            case "response.audio.delta":
                                assert isinstance(event, ResponseAudioDeltaEvent)  # nosec: B101
                                yield ResourceContent.of(
                                    event.delta,
                                    mime_type="audio/pcm",  # it seems it is a default format
                                )

                            case "response.reasoning_text.delta":
                                assert isinstance(event, ResponseReasoningTextDeltaEvent)  # nosec: B101
                                yield ModelReasoningChunk.of(
                                    TextContent.of(event.delta),
                                    meta={"kind": "reasoning_chunk"},
                                )

                            case "response.reasoning_summary_text.delta":
                                assert isinstance(event, ResponseReasoningSummaryTextDeltaEvent)  # nosec: B101
                                yield ModelReasoningChunk.of(
                                    TextContent.of(event.delta),
                                    meta={"kind": "reasoning_summary_chunk"},
                                )

                            case "response.output_item.done":
                                assert isinstance(event, ResponseOutputItemDoneEvent)  # nosec: B101
                                match event.item.type:
                                    case "reasoning":
                                        assert isinstance(event.item, ResponseReasoningItem)  # nosec: B101
                                        # final chunk with identifiers
                                        yield ModelReasoningChunk.of(
                                            TextContent.empty,
                                            meta={
                                                "kind": "reasoning",
                                                "encrypted": event.item.encrypted_content,
                                            },
                                        )

                                    case "function_call":
                                        assert isinstance(event.item, ResponseFunctionToolCall)  # nosec: B101
                                        try:
                                            yield ModelToolRequest.of(
                                                event.item.call_id,
                                                tool=event.item.name,
                                                arguments=(
                                                    json.loads(event.item.arguments)
                                                    if event.item.arguments
                                                    else {}
                                                ),
                                            )

                                        except Exception as exc:
                                            raise ModelOutputInvalid(
                                                provider="openai",
                                                model=config.model,
                                                reason=(
                                                    "Tool arguments decoding error - "
                                                    f"{type(exc).__name__}: {exc}"
                                                ),
                                            ) from exc

                                    case "image_generation_call":
                                        assert isinstance(event.item, ImageGenerationCall)  # nosec: B101
                                        if event.item.result is None:
                                            raise ModelOutputInvalid(
                                                provider="openai",
                                                model=config.model,
                                                reason="Image generation result does not contain"
                                                " an image",
                                            )

                                        yield ResourceContent.of(
                                            event.item.result,
                                            # it seems that we always get png
                                            mime_type="image/png",
                                            meta={
                                                "id": event.item.id,
                                                "status": event.item.status,
                                            },
                                        )

                                    case _:
                                        continue  # ignore other items

                            case "response.refusal.done":
                                assert isinstance(event, ResponseRefusalDoneEvent)  # nosec: B101
                                raise ModelOutputInvalid(
                                    provider="openai",
                                    model=config.model,
                                    reason=f"Response refusal: {event.refusal}",
                                )

                            case "error":
                                assert isinstance(event, ResponseErrorEvent)  # nosec: B101
                                raise ModelOutputFailed(
                                    provider="openai",
                                    model=config.model,
                                    reason=f"{event.code or 'Error'}: {event.message}",
                                )

                            case "response.completed" | "response.failed":
                                if usage := event.response.usage:
                                    record_usage_metrics(
                                        provider="openai",
                                        model=config.model,
                                        input_tokens=usage.input_tokens,
                                        cached_input_tokens=usage.input_tokens_details.cached_tokens,
                                        output_tokens=usage.output_tokens,
                                        reasoning_output_tokens=(
                                            usage.output_tokens_details.reasoning_tokens
                                        ),
                                    )

                                if error := event.response.error:
                                    raise ModelOutputFailed(
                                        provider="openai",
                                        model=config.model,
                                        reason=f"{error.code}: {error.message}",
                                    )

                                if incomplete := event.response.incomplete_details:
                                    if incomplete.reason == "max_output_tokens":
                                        raise ModelOutputLimit(
                                            provider="openai",
                                            model=config.model,
                                            max_output_tokens=unwrap_missing(
                                                config.max_output_tokens,
                                                default=0,
                                            ),
                                        )

                                    else:
                                        raise ModelOutputInvalid(
                                            provider="openai",
                                            model=config.model,
                                            reason=incomplete.reason or "incomplete",
                                        )

                            case _:
                                continue  # skip other events

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
                        "model.provider": "openai",
                        "model.name": config.model,
                        "retry_after": delay,
                    },
                )
                raise ModelRateLimit(
                    provider="openai",
                    model=config.model,
                    retry_after=delay,
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="openai",
                    model=config.model,
                    reason=str(exc),
                ) from exc


def _text_output(  # noqa: PLR0911
    output: ModelOutputSelection,
    /,
    *,
    verbosity: Literal["low", "medium", "high"] | Missing = MISSING,
) -> ResponseTextConfigParam | Omit:
    if output == "auto":
        return omit

    if output == "text":
        if verbosity is MISSING:
            return {"format": {"type": "text"}}

        return {
            "format": {"type": "text"},
            "verbosity": cast(Literal["low", "medium", "high"], verbosity),
        }

    if output == "json":
        if verbosity is MISSING:
            return {"format": {"type": "json_object"}}

        return {
            "format": {"type": "json_object"},
            "verbosity": cast(Literal["low", "medium", "high"], verbosity),
        }

    if isinstance(output, type):
        if verbosity is MISSING:
            return {
                "format": {
                    "type": "json_schema",
                    "name": output.__name__,
                    "schema": as_dict(output.__SPECIFICATION__),
                    "strict": True,
                }
            }

        return {
            "format": {
                "type": "json_schema",
                "name": output.__name__,
                "schema": as_dict(output.__SPECIFICATION__),
                "strict": True,
            },
            "verbosity": cast(Literal["low", "medium", "high"], verbosity),
        }

    # multimodal selection containing text
    if "text" in output:
        if verbosity is MISSING:
            return {"format": {"type": "text"}}

        return {
            "format": {"type": "text"},
            "verbosity": cast(Literal["low", "medium", "high"], verbosity),
        }

    return omit


def _tool_choice(
    tools: ModelTools,
    /,
) -> ToolChoiceFunctionParam | ToolChoiceOptions:
    match tools.selection:
        case "auto" | "required" | "none":
            return tools.selection

        case specification:  # specific tool declaration
            return {
                "type": "function",
                "name": specification.name,
            }


def _tools_as_tool_params(
    tools: Sequence[ModelToolSpecification],
    /,
) -> Sequence[ToolParam]:
    return [
        cast(
            ToolParam,
            FunctionToolParam(
                type="function",
                name=tool.name,
                description=tool.description or None,
                parameters=cast(dict[str, object] | None, tool.parameters)
                if tool.parameters is not None
                else {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=tool.meta.get_bool(
                    "strict_parameters",
                    default=False,
                ),
            ),
        )
        for tool in tools
    ]


def _context_to_params(
    context: ModelContext,
    /,
    vision_details: Literal["auto", "low", "high"],
) -> Generator[ResponseInputItemParam]:
    for element in context:
        if isinstance(element, ModelInput):
            yield from _model_input_to_params(
                element,
                vision_details=vision_details,
            )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            yield from _model_output_to_params(element)


def _model_input_to_params(
    element: ModelInput,
    /,
    vision_details: Literal["auto", "low", "high"],
) -> Generator[ResponseInputItemParam]:
    for block in element.input:
        if isinstance(block, MultimodalContent):
            yield Message(
                type="message",
                role="user",
                content=cast(
                    ResponseInputMessageContentListParam,
                    list(
                        _input_content_parts(
                            block,
                            vision_details=vision_details,
                        )
                    ),
                ),
            )

        else:
            yield FunctionCallOutput(
                type="function_call_output",
                call_id=block.identifier,
                output=list(
                    _input_content_parts(
                        block.result,
                        vision_details=vision_details,
                    )
                ),
            )


def _model_output_to_params(
    element: ModelOutput,
    /,
) -> Generator[ResponseInputItemParam]:
    for block in element.output:
        if isinstance(block, MultimodalContent):
            yield from _output_content_blocks(block)

        elif isinstance(block, ModelReasoning):
            match block.meta.kind:
                case "reasoning":
                    encrypted: str | None = block.meta.get_str("encrypted")
                    if not encrypted:
                        continue  # Only include reasoning when we have encrypted content.

                    yield ResponseReasoningItemParam(
                        id=block.meta.get_str(
                            "id",
                            default=f"rs_{uuid4()}",
                        ),
                        type="reasoning",
                        summary=[
                            {
                                "type": "summary_text",
                                "text": part.to_str(),
                            }
                            for part in block.reasoning.parts
                        ],
                        encrypted_content=encrypted,
                    )

                case other:
                    raise ValueError(f"Unsupported reasoning element: {other}")

        else:
            assert isinstance(block, ModelToolRequest)  # nosec: B101
            yield ResponseFunctionToolCallParam(
                type="function_call",
                call_id=block.identifier,
                name=block.tool,
                arguments=json.dumps(block.arguments),
                status="completed",
            )


def _input_content_parts(
    content: MultimodalContent,
    /,
    vision_details: Literal["auto", "low", "high"],
) -> Generator[ResponseInputTextContentParam | ResponseInputImageContentParam]:
    for part in content.parts:
        if isinstance(part, TextContent):
            yield ResponseInputTextContentParam(
                type="input_text",
                text=part.text,
            )

        elif isinstance(part, ResourceContent):
            # Only image resources are supported as OpenAI input content
            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported media - {part.mime_type}")

            yield ResponseInputImageContentParam(
                type="input_image",
                detail=vision_details,
                image_url=part.to_data_uri(),
            )

        elif isinstance(part, ResourceReference):
            # Only image references supported here; require explicit image mime
            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported media - {part.mime_type}")

            yield ResponseInputImageContentParam(
                type="input_image",
                detail=vision_details,
                image_url=part.uri,
            )

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            yield ResponseInputTextContentParam(
                type="input_text",
                text=part.to_str(),
            )


def _text_annotations_from_meta(
    meta: Meta,
) -> Generator[Annotation]:
    match meta.get("citations"):
        case None:
            pass

        case [*citations]:
            for citation in citations:
                match citation:
                    case {
                        "kind": "url_citation",
                        "url": str() as url,
                        "title": str() as title,
                        "start": int() as start_index,
                        "end": int() as end_index,
                    }:
                        yield AnnotationURLCitation(
                            type="url_citation",
                            url=url,
                            title=title,
                            start_index=start_index,
                            end_index=end_index,
                        )

                    case {
                        "kind": "container_file_citation",
                        "container_id": str() as container_id,
                        "file_id": str() as file_id,
                        "filename": str() as filename,
                        "start": int() as start_index,
                        "end": int() as end_index,
                    }:
                        yield AnnotationContainerFileCitation(
                            type="container_file_citation",
                            container_id=container_id,
                            file_id=file_id,
                            filename=filename,
                            start_index=start_index,
                            end_index=end_index,
                        )

                    case {
                        "kind": "file_citation",
                        "index": int() as index,
                        "file_id": str() as file_id,
                        "filename": str() as filename,
                    }:
                        yield AnnotationFileCitation(
                            type="file_citation",
                            file_id=file_id,
                            filename=filename,
                            index=index,
                        )

                    case {
                        "kind": "file_path",
                        "index": int() as index,
                        "file_id": str() as file_id,
                    }:
                        yield AnnotationFilePath(
                            type="file_path",
                            file_id=file_id,
                            index=index,
                        )

                    case other:
                        raise ValueError(f"Invalid citation metadata: {type(other)}")

        case other:
            raise ValueError(f"Invalid citation metadata: {type(other)}")


def _output_content_blocks(  # noqa: C901
    content: MultimodalContent,
    /,
) -> Generator[ResponseOutputMessageParam | ImageGenerationCallParam]:
    content_accumulator: list[ResponseOutputTextParam] = []

    def flush_message() -> ResponseOutputMessageParam | None:
        nonlocal content_accumulator
        if not content_accumulator:
            return None

        message = ResponseOutputMessageParam(
            id=f"msg_{uuid4()}",
            type="message",
            role="assistant",
            content=content_accumulator,
            status="completed",
        )
        content_accumulator = []

        return message

    for part in content.parts:
        if isinstance(part, TextContent):
            content_accumulator.append(
                ResponseOutputTextParam(
                    type="output_text",
                    text=part.text,
                    annotations=_text_annotations_from_meta(part.meta),
                )
            )

        elif isinstance(part, ResourceContent):
            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported media - {part.mime_type}")

            if message := flush_message():
                yield message

            yield ImageGenerationCallParam(
                id=f"img_{uuid4()}",
                type="image_generation_call",
                result=part.data,
                status="completed",
            )

        elif isinstance(part, ResourceReference):
            raise ValueError("Media is not supported as model output")

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            content_accumulator.append(
                ResponseOutputTextParam(
                    type="output_text",
                    text=part.to_str(),
                    annotations=(),
                )
            )

    if message := flush_message():
        yield message
