import json
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Final, Literal, cast
from uuid import uuid4

from haiway import MISSING, Meta, Missing, as_dict, ctx, unwrap_missing
from openai import Omit, omit
from openai import RateLimitError as OpenAIRateLimitError
from openai.types import ReasoningEffort
from openai.types.responses import (
    ResponseAudioDeltaEvent,
    ResponseErrorEvent,
    ResponseFunctionToolCall,
    ResponseInputFileContentParam,
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
from openai.types.responses.response_format_text_config_param import (
    ResponseFormatTextConfigParam,
)
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput, Message
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
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
    ModelToolSpecification,
    model_rate_limit,
    record_model_invocation,
    record_usage_metrics,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIResponsesConfig
from draive.resources import ResourceContent, ResourceReference
from draive.utils.schema import strict_schema

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
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                verbosity=config.verbosity,
                reasoning=config.reasoning,
                service_tier=config.service_tier,
                truncation=config.truncation,
            )

            # an unsupported request is prepared before the call so that it is reported
            # as itself instead of being reported as a failed generation
            text_config: ResponseTextConfigParam | Omit = _text_output(
                output,
                verbosity=config.verbosity,
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
                    reason=str(exc),
                ) from exc

            try:
                async with self._client.responses.stream(
                    model=config.model,
                    instructions=instructions or omit,
                    input=input_context,
                    tool_choice=_tool_choice(tools),
                    tools=_tools_as_tool_params(tools.specification),
                    parallel_tool_calls=True if tools else omit,
                    text=text_config,
                    reasoning=_reasoning(config),
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
                    prompt_cache_retention=unwrap_missing(
                        config.prompt_cache_retention,
                        default=omit,
                    ),
                    # responses are never stored, encrypted reasoning is what carries
                    # the reasoning state across turns without any server side copy.
                    # OpenAI compatible servers reachable through a custom base url
                    # do not implement it, requesting it there breaks the request
                    include=["reasoning.encrypted_content"] if self._base_url is None else omit,
                    # keep the whole exchange within the caller's own context
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
                                        # final chunk with identifiers - it closes the
                                        # block so the identity stays paired with the
                                        # summary text it was produced for
                                        yield ModelReasoningChunk.of(
                                            TextContent.empty,
                                            final=True,
                                            meta={
                                                "kind": "reasoning",
                                                "id": event.item.id,
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

                            # a response truncated by max_output_tokens terminates
                            # with `response.incomplete` instead of `response.completed`
                            case "response.completed" | "response.failed" | "response.incomplete":
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
                raise model_rate_limit(
                    provider="openai",
                    model=config.model,
                    retry_after=exc.response.headers.get("Retry-After"),
                ) from exc

            except ModelException as exc:
                raise exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="openai",
                    model=config.model,
                    reason=str(exc),
                ) from exc


def _text_output(
    output: ModelOutputSelection,
    /,
    *,
    verbosity: Literal["low", "medium", "high"] | Missing = MISSING,
) -> ResponseTextConfigParam | Omit:
    text_format: ResponseFormatTextConfigParam | None = _text_format(output)
    if verbosity is MISSING:
        if text_format is None:
            return omit

        return {"format": text_format}

    if text_format is None:
        return {"verbosity": cast(Literal["low", "medium", "high"], verbosity)}

    return {
        "format": text_format,
        "verbosity": cast(Literal["low", "medium", "high"], verbosity),
    }


def _text_format(
    output: ModelOutputSelection,
    /,
) -> ResponseFormatTextConfigParam | None:
    if output == "auto":
        return None

    if output == "text":
        return {"type": "text"}

    if output == "json":
        # the json_object format additionally requires the word "json" to appear
        # within the input itself, which cannot be satisfied without injecting
        # content into the request - no format is requested instead, leaving the
        # shape to the caller's own instructions. A schema backed selection maps
        # onto json_schema below and carries a real guarantee.
        return None

    if isinstance(output, type):
        schema: Mapping[str, Any] = as_dict(output.__SPECIFICATION__)
        # only strict mode makes the api enforce the schema, without it the model
        # can answer with content which fails decoding
        strict: Mapping[str, Any] | None = strict_schema(schema)
        if strict is None:
            ctx.log_debug(
                f"OpenAI strict output is unavailable for {output.__name__},"
                " the schema is delivered as a hint instead"
            )

        return {
            "type": "json_schema",
            "name": output.__name__,
            "schema": cast(dict[str, object], strict if strict is not None else schema),
            "strict": strict is not None,
        }

    # multimodal selection containing text
    if "text" in output:
        return {"type": "text"}

    # silently answering with text would not match the requested modality
    raise NotImplementedError(f"{output} output is not supported by OpenAI responses")


def _reasoning(
    config: OpenAIResponsesConfig,
    /,
) -> Reasoning | Omit:
    reasoning: Reasoning = {}
    if isinstance(config.reasoning, str):
        reasoning["effort"] = cast(ReasoningEffort, config.reasoning)
        # a summary is only ever produced along a reasoning effort
        reasoning["summary"] = config.reasoning_summary

    if isinstance(config.reasoning_context, str):
        reasoning["context"] = config.reasoning_context

    if isinstance(config.reasoning_mode, str):
        reasoning["mode"] = config.reasoning_mode

    return reasoning or omit


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
    return [_tool_as_tool_param(tool) for tool in tools]


def _tool_as_tool_param(
    tool: ModelToolSpecification,
    /,
) -> ToolParam:
    parameters: Mapping[str, Any] = (
        tool.parameters
        if tool.parameters is not None
        else {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    strict: bool = tool.meta.get_bool(
        "strict_parameters",
        default=False,
    )
    if strict:
        # strict mode demands every argument, a schema left as declared is rejected
        # as soon as any argument carries a default. A mapping of free form keys is
        # stripped out of an enforced argument object, so a schema holding one has to
        # go unenforced rather than lose whatever the model puts there.
        converted: Mapping[str, Any] | None = strict_schema(
            parameters,
            open_maps=False,
        )
        if converted is not None:
            parameters = converted

        else:
            ctx.log_debug(
                f"OpenAI strict parameters are unavailable for the {tool.name} tool,"
                " its arguments are delivered without enforcement instead"
            )
            strict = False

    return cast(
        ToolParam,
        FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description or None,
            parameters=cast(dict[str, object], parameters),
            strict=strict,
        ),
    )


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
            yield from _model_output_to_params(
                element,
                vision_details=vision_details,
            )


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
                        block.content,
                        vision_details=vision_details,
                    )
                ),
            )


def _model_output_to_params(
    element: ModelOutput,
    /,
    vision_details: Literal["auto", "low", "high"],
) -> Generator[ResponseInputItemParam]:
    for block in element.output:
        if isinstance(block, MultimodalContent):
            yield from _output_content_blocks(
                block,
                vision_details=vision_details,
            )

        elif isinstance(block, ModelReasoning):
            match block.meta.kind:
                case "reasoning":
                    encrypted: str | None = block.meta.get_str("encrypted")
                    item_id: str | None = block.meta.get_str("id")
                    # Encrypted content is cryptographically bound to its original
                    # item id; replaying it with a fabricated id is rejected by the API.
                    if not encrypted or not item_id:
                        continue  # Only include reasoning when we have encrypted content and id.

                    yield ResponseReasoningItemParam(
                        id=item_id,
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


# non image resources are delivered as files, the api takes documents and plain text
_FILE_MIME_TYPES: Final[Mapping[str, str]] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
    "application/json": "json",
}


def _input_content_parts(
    content: MultimodalContent,
    /,
    vision_details: Literal["auto", "low", "high"],
) -> Generator[
    ResponseInputTextContentParam | ResponseInputImageContentParam | ResponseInputFileContentParam
]:
    for part in content.parts:
        if isinstance(part, TextContent):
            yield ResponseInputTextContentParam(
                type="input_text",
                text=part.text,
            )

        elif isinstance(part, ResourceContent):
            if extension := _FILE_MIME_TYPES.get(part.mime_type):
                # the api requires a filename to resolve the file kind
                yield ResponseInputFileContentParam(
                    type="input_file",
                    filename=f"file.{extension}",
                    file_data=part.to_data_uri(),
                )
                continue

            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported media - {part.mime_type}")

            yield ResponseInputImageContentParam(
                type="input_image",
                detail=vision_details,
                image_url=part.to_data_uri(),
            )

        elif isinstance(part, ResourceReference):
            if part.mime_type in _FILE_MIME_TYPES:
                yield ResponseInputFileContentParam(
                    type="input_file",
                    file_url=part.uri,
                )
                continue

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
    vision_details: Literal["auto", "low", "high"],
) -> Generator[ResponseOutputMessageParam | Message]:
    text_accumulator: list[ResponseOutputTextParam] = []
    image_accumulator: list[ResponseInputImageContentParam] = []

    def flush_text() -> ResponseOutputMessageParam | None:
        nonlocal text_accumulator
        if not text_accumulator:
            return None

        message = ResponseOutputMessageParam(
            id=f"msg_{uuid4()}",
            type="message",
            role="assistant",
            content=text_accumulator,
            status="completed",
        )
        text_accumulator = []

        return message

    def flush_images() -> Message | None:
        # An assistant message accepts only text and refusal parts, while replaying an
        # `image_generation_call` requires the id of a server side item which never
        # exists for an unstored response. A user message carrying the image is the
        # only representation the api accepts for an image produced within a turn.
        nonlocal image_accumulator
        if not image_accumulator:
            return None

        message = Message(
            type="message",
            role="user",
            content=cast(ResponseInputMessageContentListParam, image_accumulator),
        )
        image_accumulator = []

        return message

    for part in content.parts:
        if isinstance(part, TextContent):
            if message := flush_images():
                yield message

            text_accumulator.append(
                ResponseOutputTextParam(
                    type="output_text",
                    text=part.text,
                    annotations=list(_text_annotations_from_meta(part.meta)),
                )
            )

        elif isinstance(part, ResourceContent):
            if not part.mime_type.startswith("image"):
                raise ValueError(f"Unsupported media - {part.mime_type}")

            if message := flush_text():
                yield message

            image_accumulator.append(
                ResponseInputImageContentParam(
                    type="input_image",
                    detail=vision_details,
                    image_url=part.to_data_uri(),
                )
            )

        elif isinstance(part, ResourceReference):
            raise ValueError("Media is not supported as model output")

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            if message := flush_images():
                yield message

            text_accumulator.append(
                ResponseOutputTextParam(
                    type="output_text",
                    text=part.to_str(),
                    annotations=(),
                )
            )

    if message := flush_text():
        yield message

    if message := flush_images():
        yield message
