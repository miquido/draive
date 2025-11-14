import json
import random
from collections.abc import AsyncGenerator, Collection, Coroutine, Generator, Mapping, Sequence
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import MISSING, Meta, Missing, as_dict, ctx, unwrap_missing
from openai import Omit, omit
from openai import RateLimitError as OpenAIRateLimitError
from openai.types.responses import (
    Response,
    ResponseAudioDeltaEvent,
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInputImageContentParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextContentParam,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseOutputText,
    ResponseOutputTextParam,
    ResponseReasoningItem,
    ResponseReasoningItemParam,
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
from openai.types.responses.response_output_item_done_event import ResponseOutputItemDoneEvent
from openai.types.responses.response_output_text import (
    Annotation as ResponseTextAnnotation,
)
from openai.types.responses.response_output_text import (
    AnnotationContainerFileCitation as ResponseTextAnnotationContainerFileCitation,
)
from openai.types.responses.response_output_text import (
    AnnotationFileCitation as ResponseTextAnnotationFileCitation,
)
from openai.types.responses.response_output_text import (
    AnnotationFilePath as ResponseTextAnnotationFilePath,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponseTextAnnotationURLCitation,
)
from openai.types.responses.response_output_text_param import (
    Annotation,
    AnnotationContainerFileCitation,
    AnnotationFileCitation,
    AnnotationFilePath,
    AnnotationURLCitation,
)
from openai.types.shared_params.reasoning import Reasoning

from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputInvalid,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelRateLimit,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolsDeclaration,
    ModelToolSpecification,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIResponsesConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIResponses",)

# Consistent randomized backoff window for rate limits (seconds)
RATE_LIMIT_RETRY_RANGE: tuple[float, float] = (0.3, 3.0)


class OpenAIResponses(OpenAIAPI):
    def generative_model(self) -> GenerativeModel:
        return GenerativeModel(generating=self.completion)

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        stream: Literal[False] = False,
        config: OpenAIResponsesConfig | None = None,
        cache_key: str | None = None,
        **extra: Any,
    ) -> Coroutine[None, None, ModelOutput]: ...

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        stream: Literal[True],
        config: OpenAIResponsesConfig | None = None,
        cache_key: str | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    def completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        stream: bool = False,
        config: OpenAIResponsesConfig | None = None,
        cache_key: str | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config if config is not None else ctx.state(OpenAIResponsesConfig),
                cache_key=cache_key,
                **extra,
            )

        else:
            return self._completion(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config if config is not None else ctx.state(OpenAIResponsesConfig),
                cache_key=cache_key,
                **extra,
            )

    async def _completion(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        config: OpenAIResponsesConfig,
        cache_key: str | None = None,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record_info(
                attributes={
                    "model.provider": "openai",
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.output": str(output),
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
                    "model.cache_key": cache_key,
                    "model.stream": False,
                },
            )
            ctx.record_debug(
                attributes={
                    "model.instructions": instructions,
                    "model.tools": [tool.name for tool in tools.specifications],
                    "model.context": [element.to_str() for element in context],
                },
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
                response: Response = await self._client.responses.create(
                    model=config.model,
                    instructions=instructions or omit,
                    input=input_context,
                    temperature=unwrap_missing(config.temperature, default=omit),
                    tool_choice=_tool_choice(tools),
                    tools=_tools_as_tool_params(tools.specifications),
                    text=_text_output(output, verbosity=config.verbosity),
                    reasoning=(
                        Reasoning(
                            effort=config.reasoning,
                            summary=config.reasoning_summary,
                        )
                        if isinstance(config.reasoning, str)
                        else omit
                    ),
                    parallel_tool_calls=config.parallel_tool_calls,
                    max_output_tokens=config.max_output_tokens,
                    service_tier=config.service_tier,
                    truncation=config.truncation,
                    safety_identifier=config.safety_identifier or omit,
                    prompt_cache_key=cache_key or omit,
                    include=["reasoning.encrypted_content"]
                    # for gpt-5 model family we need to request encrypted reasoning
                    if "gpt-5" in config.model.lower()
                    else omit,
                    store=False,
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
                    provider="openai",
                    model=config.model,
                    retry_after=delay,
                ) from exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="openai",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            if usage := response.usage:
                ctx.record_info(
                    metric="model.input_tokens",
                    value=getattr(usage, "input_tokens", 0),
                    unit="tokens",
                    kind="counter",
                    attributes={
                        "model.provider": "openai",
                        "model.name": config.model,
                    },
                )
                ctx.record_info(
                    metric="model.output_tokens",
                    value=getattr(usage, "output_tokens", 0),
                    unit="tokens",
                    kind="counter",
                    attributes={
                        "model.provider": "openai",
                        "model.name": config.model,
                    },
                )

            if error := response.error:
                raise ModelOutputFailed(
                    provider="openai",
                    model=config.model,
                    reason=f"{error.code}: {error.message}",
                )

            if not response.output:
                raise ModelOutputFailed(
                    provider="openai",
                    model=config.model,
                    reason="Missing output in response",
                )

            output_blocks: list[ModelOutputBlock] = []
            for block in response.output:
                if isinstance(block, ResponseOutputMessage):
                    for part in block.content:
                        if isinstance(part, ResponseOutputText):
                            output_blocks.append(
                                MultimodalContent.of(
                                    TextContent.of(
                                        text=part.text,
                                        meta=_meta_from_text_annotations(part.annotations),
                                    )
                                )
                            )

                        else:
                            raise ModelOutputInvalid(
                                provider="openai",
                                model=config.model,
                                reason=f"Response refusal: {part.refusal}",
                            )

                elif isinstance(block, ResponseFunctionToolCall):
                    try:
                        output_blocks.append(
                            ModelToolRequest.of(
                                block.call_id,
                                tool=block.name,
                                arguments=json.loads(block.arguments),
                            )
                        )

                    except Exception as exc:
                        raise ModelOutputInvalid(
                            provider="openai",
                            model=config.model,
                            reason=f"Tool arguments decoding error - {type(exc).__name__}: {exc}",
                        ) from exc

                elif isinstance(block, ResponseReasoningItem):
                    output_blocks.append(
                        ModelReasoning.of(
                            MultimodalContent.of(
                                *(TextContent(text=content.text) for content in block.summary),
                            ),
                            meta={
                                "id": block.id,
                                "kind": "reasoning",
                                "encrypted": block.encrypted_content,
                            },
                        )
                    )

                elif isinstance(block, ImageGenerationCall):
                    if block.result is None:
                        raise ModelOutputInvalid(
                            provider="openai",
                            model=config.model,
                            reason="Image generation result does not contain an image",
                        )

                    output_blocks.append(
                        MultimodalContent.of(
                            ResourceContent.of(
                                block.result,
                                mime_type="image/png",  # it seems that we always get png
                            ),
                        )
                    )

                else:
                    raise ModelOutputInvalid(
                        provider="openai",
                        model=config.model,
                        reason=f"Unsupported response block: {type(block).__name__}",
                    )

            model_output: ModelOutput = ModelOutput.of(
                *output_blocks,
                meta={
                    "identifier": response.id,
                    "model": config.model,
                },
            )

            if incomplete := response.incomplete_details:
                if incomplete.reason == "max_output_tokens":
                    raise ModelOutputLimit(
                        provider="openai",
                        model=config.model,
                        max_output_tokens=config.max_output_tokens or 0,
                        content=output_blocks,
                    )

                else:
                    raise ModelOutputInvalid(
                        provider="openai",
                        model=config.model,
                        reason=incomplete.reason or "incomplete",
                    )

            return model_output

    async def _completion_stream(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        config: OpenAIResponsesConfig,
        cache_key: str | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        async with ctx.scope("model.completion.stream"):
            ctx.record_info(
                attributes={
                    "model.provider": "openai",
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.output": str(output),
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
                    "model.cache_key": cache_key,
                    "model.stream": True,
                },
            )
            ctx.record_debug(
                attributes={
                    "model.instructions": instructions,
                    "model.tools": [tool.name for tool in tools.specifications],
                    "model.context": [element.to_str() for element in context],
                },
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
                    temperature=unwrap_missing(config.temperature, default=omit),
                    tool_choice=_tool_choice(tools),
                    tools=_tools_as_tool_params(tools.specifications),
                    text=_text_output(output, verbosity=config.verbosity),
                    reasoning=(
                        Reasoning(
                            effort=config.reasoning,
                            summary=config.reasoning_summary,
                        )
                        if isinstance(config.reasoning, str)
                        else omit
                    ),
                    parallel_tool_calls=config.parallel_tool_calls,
                    max_output_tokens=config.max_output_tokens,
                    service_tier=config.service_tier,
                    truncation=config.truncation,
                    safety_identifier=config.safety_identifier or omit,
                    prompt_cache_key=cache_key or omit,
                    include=["reasoning.encrypted_content"]
                    # for gpt-5 model family we need to request encrypted reasoning
                    if "gpt-5" in config.model.lower()
                    else omit,
                    store=False,
                ) as stream:
                    async for event in stream:
                        if isinstance(event, ResponseTextDeltaEvent):
                            yield TextContent(text=event.delta)

                        elif isinstance(event, ResponseAudioDeltaEvent):
                            yield ResourceContent.of(
                                event.delta,
                                mime_type="audio/pcm16",  # it seems it is a default format
                            )

                        elif isinstance(event, ResponseReasoningTextDeltaEvent):
                            yield ModelReasoning.of(
                                event.delta,
                                meta={"kind": "reasoning_chunk"},
                            )

                        elif isinstance(event, ResponseOutputItemDoneEvent):
                            if isinstance(event.item, ResponseFunctionToolCall):
                                try:
                                    # arguments in DoneEvent should be a complete JSON string
                                    args: Mapping[str, Any] = (
                                        json.loads(event.item.arguments)
                                        if event.item.arguments
                                        else {}
                                    )
                                    yield ModelToolRequest.of(
                                        event.item.call_id,
                                        tool=event.item.name,
                                        arguments=args,
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

                            elif isinstance(event.item, ImageGenerationCall):
                                if event.item.result is None:
                                    raise ModelOutputInvalid(
                                        provider="openai",
                                        model=config.model,
                                        reason="Image generation result does not contain an image",
                                    )

                                yield ResourceContent.of(
                                    event.item.result,
                                    mime_type="image/png",  # it seems that we always get png
                                )

                            elif isinstance(event.item, ResponseReasoningItem):
                                yield ModelReasoning.of(
                                    MultimodalContent.of(
                                        *(
                                            TextContent(text=content.text)
                                            for content in event.item.summary
                                        ),
                                    ),
                                    meta={
                                        "id": event.item.id,
                                        "kind": "reasoning",
                                        "encrypted": event.item.encrypted_content,
                                    },
                                )

                        elif isinstance(event, ResponseRefusalDoneEvent):
                            raise ModelOutputInvalid(
                                provider="openai",
                                model=config.model,
                                reason=f"Response refusal: {event.refusal}",
                            )

                        elif isinstance(event, ResponseErrorEvent):
                            raise ModelOutputFailed(
                                provider="openai",
                                model=config.model,
                                reason=f"{event.code or 'Error'}: {event.message}",
                            )

                        elif isinstance(
                            event,
                            ResponseCompletedEvent | ResponseFailedEvent | ResponseIncompleteEvent,
                        ):
                            if usage := event.response.usage:
                                ctx.record_info(
                                    metric="model.input_tokens",
                                    value=getattr(usage, "input_tokens", 0),
                                    unit="tokens",
                                    kind="counter",
                                    attributes={
                                        "model.provider": "openai",
                                        "model.name": config.model,
                                    },
                                )
                                ctx.record_info(
                                    metric="model.output_tokens",
                                    value=getattr(usage, "output_tokens", 0),
                                    unit="tokens",
                                    kind="counter",
                                    attributes={
                                        "model.provider": "openai",
                                        "model.name": config.model,
                                    },
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
                                        max_output_tokens=config.max_output_tokens or 0,
                                        content=(),  # we have already streamed content
                                    )

                                else:
                                    raise ModelOutputInvalid(
                                        provider="openai",
                                        model=config.model,
                                        reason=incomplete.reason or "incomplete",
                                    )

                        else:
                            continue  # skip other events

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
    tools: ModelToolsDeclaration,
    /,
) -> ToolChoiceFunctionParam | ToolChoiceOptions:
    match tools.selection:
        case "auto" | "required" | "none":
            return tools.selection

        case specification:  # specific tool name
            return {
                "type": "function",
                "name": specification,
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
                or {
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
        try:
            if isinstance(element, ModelInput):
                yield from _model_input_to_params(
                    element,
                    vision_details=vision_details,
                )

            else:
                assert isinstance(element, ModelOutput)  # nosec: B101
                yield from _model_output_to_params(element)

        except Exception:
            ctx.log_debug(f"context_to_params error on element: {type(element).__name__}")
            raise


def _model_input_to_params(
    element: ModelInput,
    /,
    vision_details: Literal["auto", "low", "high"],
) -> Generator[ResponseInputItemParam]:
    for block in element.blocks:
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
) -> Generator[ResponseInputItemParam]:
    for block in element.blocks:
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
                            default=f"rs_{uuid4().hex}",
                        ),
                        type="reasoning",
                        summary=[
                            {
                                "type": "summary_text",
                                "text": part.to_str(),
                            }
                            for part in block.content.parts
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
            if part.mime_type is None or not part.mime_type.startswith("image"):
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
                text=part.artifact.to_str(),
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


def _meta_from_text_annotations(
    annotations: Collection[ResponseTextAnnotation] | None,
) -> Meta | None:
    if not annotations:
        return None

    citations: list[dict[str, str | int]] = []
    for annotation in annotations:
        if isinstance(annotation, ResponseTextAnnotationURLCitation):
            citations.append(
                {
                    "kind": "url_citation",
                    "url": annotation.url,
                    "title": (annotation.title or ""),
                    "start": int(annotation.start_index),
                    "end": int(annotation.end_index),
                }
            )

        elif isinstance(annotation, ResponseTextAnnotationContainerFileCitation):
            citations.append(
                {
                    "kind": "container_file_citation",
                    "container_id": annotation.container_id,
                    "file_id": annotation.file_id,
                    "filename": annotation.filename,
                    "start": int(annotation.start_index),
                    "end": int(annotation.end_index),
                }
            )

        elif isinstance(annotation, ResponseTextAnnotationFileCitation):
            citations.append(
                {
                    "kind": "file_citation",
                    "file_id": annotation.file_id,
                    "filename": annotation.filename,
                    "index": int(annotation.index),
                }
            )

        else:
            assert isinstance(annotation, ResponseTextAnnotationFilePath)  # nosec: B101
            citations.append(
                {
                    "kind": "file_path",
                    "file_id": annotation.file_id,
                    "index": int(annotation.index),
                }
            )

    if not citations:
        return None

    return Meta.of({"citations": citations})


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
            id=f"msg_{uuid4().hex}",
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
                id=f"img_{uuid4().hex}",
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
                    text=part.artifact.to_str(),
                    annotations=(),
                )
            )

    if message := flush_message():
        yield message
