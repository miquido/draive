import random
from base64 import b64decode, b64encode, urlsafe_b64decode
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
    Sequence,
)
from itertools import chain
from typing import Any, Literal, cast, overload
from uuid import uuid4

from google.api_core.exceptions import ResourceExhausted  # pyright: ignore[reportMissingImport]
from google.genai.types import (
    Candidate,
    ContentListUnionDict,  # pyright: ignore[reportUnknownVariableType]
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
    Part,
    PartDict,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import META_EMPTY, MISSING, as_dict, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import (
    GeminiConfig,
    GeminiSafetyConfig,
)
from draive.gemini.utils import unwrap_missing
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelInputBlocks,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputBlocks,
    ModelOutputFailed,
    ModelOutputLimit,
    ModelOutputSelection,
    ModelRateLimit,
    ModelReasoning,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolResponse,
    ModelToolsDeclaration,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.parameters.model import DataModel
from draive.resources import ResourceContent, ResourceReference

__all__ = ("GeminiGenerating",)

# Consistent randomized backoff window for rate limits (seconds)
RATE_LIMIT_RETRY_RANGE: tuple[float, float] = (0.3, 3.0)


class GeminiGenerating(GeminiAPI):
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
        config: GeminiConfig | None = None,
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
        config: GeminiConfig | None = None,
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
        config: GeminiConfig | None = None,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | Coroutine[None, None, ModelOutput]:
        if stream:
            return self._completion_stream(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config or ctx.state(GeminiConfig),
                **extra,
            )

        else:
            return self._completion(
                instructions=instructions,
                context=context,
                tools=tools,
                output=output,
                config=config or ctx.state(GeminiConfig),
                **extra,
            )

    async def _completion(
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        config: GeminiConfig,
        **extra: Any,
    ) -> ModelOutput:
        async with ctx.scope("model.completion"):
            ctx.record_info(
                attributes={
                    "model.provider": "gemini",
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.output": str(output),
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.thinking_budget": config.thinking_budget,
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
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

            request_config: GenerateContentConfigDict = _prepare_request_config(
                instructions=instructions,
                tools=tools,
                output=output,
                config=config,
            )
            request_content: ContentListUnionDict = [  # pyright: ignore[reportUnknownVariableType]
                _context_element_as_content(element) for element in context
            ]

            try:
                completion: GenerateContentResponse = (
                    await self._client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
                        model=config.model,
                        config=request_config,
                        contents=request_content,
                    )
                )

            except ResourceExhausted as exc:
                ctx.record_warning(
                    event="model.rate_limit",
                    attributes={
                        "model.provider": "gemini",
                        "model.name": config.model,
                    },
                )
                raise ModelRateLimit(
                    provider="gemini",
                    model=config.model,
                    retry_after=random.uniform(*RATE_LIMIT_RETRY_RANGE),  # nosec: B311
                ) from exc

            except Exception as exc:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            _record_usage_metrics(
                completion.usage_metadata,
                model=config.model,
            )

            if not completion.candidates:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason="Missing candidates in response",
                )

            completion_candidate: Candidate = completion.candidates[0]  # we always request only one

            if completion_candidate.safety_ratings:
                ctx.record_info(
                    event="model.safety.results",
                    attributes={
                        "results": [
                            f"{rating.category} |blocked: {rating.blocked}"
                            f" |probability:{rating.probability_score}"
                            f" |severity:{rating.severity_score}"
                            for rating in completion_candidate.safety_ratings
                            if rating.category
                        ],
                    },
                )

            if completion_candidate.finish_reason == FinishReason.SAFETY:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=f"Safety filtering: {completion_candidate.finish_message}",
                )

            if completion_candidate.finish_reason not in (
                FinishReason.STOP,
                FinishReason.MAX_TOKENS,
            ):
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=f"Completion error: {completion_candidate.finish_message}",
                )

            if not completion_candidate.content or not completion_candidate.content.parts:
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason="Empty completion content",
                )

            output_blocks: Iterable[ModelOutputBlock] = chain.from_iterable(
                _part_as_output_blocks(part) for part in completion_candidate.content.parts
            )

            if completion_candidate.finish_reason == FinishReason.MAX_TOKENS:
                # include the content collected so far and any tool requests
                raise ModelOutputLimit(
                    provider="gemini",
                    model=config.model,
                    max_output_tokens=unwrap_missing(config.max_output_tokens, default=0),
                    content=tuple(output_blocks),
                )

            return ModelOutput.of(
                *output_blocks,
                meta={
                    "identifier": completion.response_id,
                    "model": config.model,
                    "finish_reason": str(completion_candidate.finish_reason),
                },
            )

    async def _completion_stream(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        context: ModelContext,
        tools: ModelToolsDeclaration,
        output: ModelOutputSelection,
        config: GeminiConfig,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        async with ctx.scope("model.completion.stream"):
            ctx.record_info(
                attributes={
                    "model.provider": "gemini",
                    "model.name": config.model,
                    "model.temperature": config.temperature,
                    "model.output": str(output),
                    "model.max_output_tokens": config.max_output_tokens,
                    "model.thinking_budget": config.thinking_budget,
                    "model.tools.count": len(tools.specifications),
                    "model.tools.selection": tools.selection,
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

            # Build request same as non-streaming
            request_config: GenerateContentConfigDict = _prepare_request_config(
                instructions=instructions,
                tools=tools,
                output=output,
                config=config,
            )
            request_content: ContentListUnionDict = [  # pyright: ignore[reportUnknownVariableType]
                _context_element_as_content(element) for element in context
            ]

            last_usage_meta: GenerateContentResponseUsageMetadata | None = None
            try:
                response_stream: AsyncIterable[
                    GenerateContentResponse
                ] = await self._client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
                    model=config.model,
                    config=request_config,
                    contents=request_content,
                )

                collected_blocks: list[ModelOutputBlock] = []
                finish_reason: FinishReason | None = None
                finish_message: str | None = None

                async for chunk in response_stream:
                    if chunk.usage_metadata:  # chunks provide usage summary instead of delta
                        last_usage_meta = chunk.usage_metadata

                    if not chunk.candidates:
                        continue

                    chunk_candidate: Candidate = chunk.candidates[0]  # we always request only one
                    finish_reason = chunk_candidate.finish_reason
                    finish_message = chunk_candidate.finish_message

                    if chunk_candidate.safety_ratings:
                        ctx.record_info(
                            event="model.safety.results",
                            attributes={
                                "results": [
                                    f"{rating.category} |blocked: {rating.blocked}"
                                    f" |probability:{rating.probability_score}"
                                    f" |severity:{rating.severity_score}"
                                    for rating in chunk_candidate.safety_ratings
                                    if rating.category
                                ],
                            },
                        )

                    if not chunk_candidate.content or not chunk_candidate.content.parts:
                        continue

                    for part in chunk_candidate.content.parts:
                        collected_blocks.extend(_part_as_output_blocks(part))
                        for element in _part_as_stream_elements(part):
                            # element is either a MultimodalContentElement or ModelToolRequest
                            yield element

                if finish_reason == FinishReason.SAFETY:
                    raise ModelOutputFailed(
                        provider="gemini",
                        model=config.model,
                        reason=f"Safety filtering: {finish_message or ''}",
                    )

                if finish_reason == FinishReason.MAX_TOKENS:
                    raise ModelOutputLimit(
                        provider="gemini",
                        model=config.model,
                        max_output_tokens=unwrap_missing(
                            config.max_output_tokens,
                            default=0,
                        ),
                        content=tuple(collected_blocks),
                    )

                if finish_reason not in (None, FinishReason.STOP):
                    raise ModelOutputFailed(
                        provider="gemini",
                        model=config.model,
                        reason=(
                            f"Completion error: {finish_message}"
                            if finish_message
                            else "Completion error"
                        ),
                    )

            except ResourceExhausted as exc:
                ctx.record_warning(
                    event="model.rate_limit",
                    attributes={
                        "model.provider": "gemini",
                        "model.name": config.model,
                    },
                )
                # Propagate as ModelRateLimit with randomized backoff window
                raise ModelRateLimit(
                    provider="gemini",
                    model=config.model,
                    retry_after=random.uniform(*RATE_LIMIT_RETRY_RANGE),  # nosec: B311
                ) from exc

            except Exception as exc:
                # Convert to ModelOutputFailed for consistency
                raise ModelOutputFailed(
                    provider="gemini",
                    model=config.model,
                    reason=str(exc),
                ) from exc

            finally:
                _record_usage_metrics(
                    last_usage_meta,
                    model=config.model,
                )


def _record_usage_metrics(
    usage: GenerateContentResponseUsageMetadata | None,
    *,
    model: str,
) -> None:
    if usage is None:
        return

    ctx.record_info(
        metric="model.input_tokens",
        value=usage.prompt_token_count or 0,
        unit="tokens",
        kind="counter",
        attributes={
            "model.provider": "gemini",
            "model.name": model,
        },
    )
    ctx.record_info(
        metric="model.input_tokens.cached",
        value=usage.cached_content_token_count or 0,
        unit="tokens",
        kind="counter",
        attributes={
            "model.provider": "gemini",
            "model.name": model,
        },
    )
    ctx.record_info(
        metric="model.output_tokens",
        value=usage.candidates_token_count or 0,
        unit="tokens",
        kind="counter",
        attributes={
            "model.provider": "gemini",
            "model.name": model,
        },
    )


def _prepare_request_config(  # noqa: C901, PLR0912, PLR0915
    *,
    instructions: ModelInstructions,
    tools: ModelToolsDeclaration,
    output: ModelOutputSelection,
    config: GeminiConfig,
) -> GenerateContentConfigDict:
    temperature: float | None = unwrap_missing(config.temperature)
    max_output_tokens: int | None = unwrap_missing(config.max_output_tokens)
    seed: int | None = unwrap_missing(config.seed)

    response_schema: SchemaDict | None
    response_modalities: list[Modality] | None
    response_mime_type: str | None

    # Prefer explicit isinstance check for structured output to satisfy typing
    if isinstance(output, type):
        response_schema = _normalized_schema(output)
        response_modalities = [Modality.TEXT]
        response_mime_type = "application/json"

    else:
        match output:
            case "auto":
                # not specified at all - use defaults
                response_schema = None
                response_modalities = None
                response_mime_type = None

            case "text":
                response_schema = None
                response_modalities = [Modality.TEXT]
                response_mime_type = "text/plain"

            case "json":
                response_schema = None
                response_modalities = [Modality.TEXT]
                response_mime_type = "application/json"

            case "image":
                response_schema = None
                response_modalities = [
                    Modality.TEXT,
                    Modality.IMAGE,
                ]  # google api does not allow to specify only image
                response_mime_type = None

            case "audio":
                response_schema = None
                response_modalities = [Modality.AUDIO]
                response_mime_type = None

            case "video":
                raise NotImplementedError("video output is not supported by Gemini")

            case ["text", "image"] | ["image", "text"]:
                response_schema = None
                response_modalities = [
                    Modality.TEXT,
                    Modality.IMAGE,
                ]
                response_mime_type = None

            case other:
                raise NotImplementedError(f"{other} output is not supported by Gemini")

    functions: list[FunctionDeclarationDict] | None
    function_calling_mode: FunctionCallingConfigMode | None
    allowed_function_names: list[str] | None = None
    if tools.specifications:
        functions = [
            FunctionDeclarationDict(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=cast(SchemaDict, tool.parameters),
            )
            for tool in tools.specifications or []
        ]
        match tools.selection:
            case "auto":
                function_calling_mode = FunctionCallingConfigMode.AUTO

            case "required":
                function_calling_mode = FunctionCallingConfigMode.ANY

            case "none":
                functions = None  # no need to pass functions if none can be used
                function_calling_mode = FunctionCallingConfigMode.NONE

            case specific_tool:  # handle specific tool selection
                function_calling_mode = FunctionCallingConfigMode.ANY
                allowed_function_names = [specific_tool]

    else:  # no functions
        functions = None
        function_calling_mode = FunctionCallingConfigMode.NONE

    media_resolution: MediaResolution | None
    match config.media_resolution:
        case "low":
            media_resolution = MediaResolution.MEDIA_RESOLUTION_LOW

        case "medium":
            media_resolution = MediaResolution.MEDIA_RESOLUTION_MEDIUM

        case "high":
            media_resolution = MediaResolution.MEDIA_RESOLUTION_HIGH

        case _:
            media_resolution = None

    speech_config: SpeechConfigDict | None
    if config.speech_voice_name:
        speech_config = {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": cast(str, config.speech_voice_name),
                },
            }
        }

    else:
        speech_config = None

    safety_config: GeminiSafetyConfig = unwrap_missing(
        config.safety,
        default=GeminiSafetyConfig(),
    )

    return {
        "temperature": temperature,
        "top_p": unwrap_missing(config.top_p),
        "top_k": unwrap_missing(config.top_k),
        "max_output_tokens": max_output_tokens,
        "candidate_count": 1,
        "seed": seed,
        "stop_sequences": unwrap_missing(
            config.stop_sequences,
            transform=as_list,
        ),
        "safety_settings": [
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold(safety_config.harm_category_hate_speech_threshold),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold(
                    safety_config.harm_category_dangerous_content_threshold
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold(safety_config.harm_category_harassment_threshold),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold(
                    safety_config.harm_category_sexually_explicit_threshold
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                "threshold": HarmBlockThreshold(
                    safety_config.harm_category_civic_integrity_threshold
                ),
            },
        ],
        "system_instruction": instructions,
        "tools": [{"function_declarations": functions}] if functions else None,
        "tool_config": {
            "function_calling_config": {
                "mode": function_calling_mode,
                **(
                    {"allowed_function_names": allowed_function_names}
                    if allowed_function_names
                    else {}
                ),
            }
        }
        if function_calling_mode
        else None,
        # prevent google from automatically resolving function calls
        "automatic_function_calling": {
            "disable": True,
            "maximum_remote_calls": None,
        }
        if function_calling_mode
        else None,
        "media_resolution": media_resolution,
        "response_modalities": response_modalities,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
        "speech_config": speech_config,
        "thinking_config": {
            "include_thoughts": (cast(int, config.thinking_budget) > 0),
            "thinking_budget": config.thinking_budget,
        }
        if config.thinking_budget is not MISSING
        else None,
    }


def _context_element_as_content(
    element: ModelContextElement,
) -> dict[str, Any]:
    if isinstance(element, ModelInput):
        return {
            "role": "user",
            "parts": [*block_parts(element.blocks)],
        }

    else:
        assert isinstance(element, ModelOutput)  # nosec: B101
        return {
            "role": "model",
            "parts": [*block_parts(element.blocks)],
        }


def _part_as_output_blocks(
    part: Part,
) -> Generator[ModelOutputBlock]:
    if part.text:
        # assuming only text thinking is possible
        if part.thought:
            yield ModelReasoning.of(
                part.text,
                meta={
                    "kind": "thought",
                    "signature": b64encode(part.thought_signature).decode()
                    if part.thought_signature
                    else None,
                },
            )

        else:
            yield MultimodalContent.of(TextContent.of(part.text))

    if part.function_call and part.function_call.name:
        yield ModelToolRequest(
            identifier=part.function_call.id or uuid4().hex,
            tool=part.function_call.name,
            arguments=part.function_call.args if part.function_call.args is not None else {},
            meta=META_EMPTY,
        )

    if part.inline_data and part.inline_data.data:  # there is no content without content...
        yield MultimodalContent.of(
            ResourceContent.of(
                part.inline_data.data,
                mime_type=part.inline_data.mime_type or "application/octet-stream",
            )
        )

    if part.file_data and part.file_data.file_uri:  # there is no content without content...
        yield MultimodalContent.of(
            ResourceReference.of(
                part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )


def _part_as_stream_elements(
    part: Part,
) -> Generator[ModelStreamOutput]:
    if part.text:
        if part.thought:
            yield ModelReasoning.of(
                part.text,
                meta={
                    "kind": "thought",
                    "signature": b64encode(part.thought_signature).decode()
                    if part.thought_signature
                    else None,
                },
            )

        else:
            yield TextContent.of(part.text)

    if part.function_call and part.function_call.name:
        yield ModelToolRequest(
            identifier=part.function_call.id or uuid4().hex,
            tool=part.function_call.name,
            arguments=part.function_call.args if part.function_call.args is not None else {},
            meta=META_EMPTY,
        )

    if part.inline_data and part.inline_data.data:  # there is no content without content...
        yield ResourceContent.of(
            part.inline_data.data,
            mime_type=part.inline_data.mime_type or "application/octet-stream",
        )

    if part.file_data and part.file_data.file_uri:  # there is no content without content...
        yield ResourceReference.of(
            part.file_data.file_uri,
            mime_type=part.file_data.mime_type,
        )


def block_parts(
    blocks: ModelInputBlocks | ModelOutputBlocks,
    /,
) -> Generator[PartDict]:
    for block in blocks:
        if isinstance(block, ModelToolRequest):
            yield {
                "function_call": {
                    "id": block.identifier,
                    "name": block.tool,
                    "args": as_dict(block.arguments),
                }
            }

        elif isinstance(block, ModelToolResponse):
            yield {
                "function_response": {
                    "id": block.identifier,
                    "name": block.tool,
                    "response": {
                        "error": list(content_parts(block.content)),
                    }
                    if block.handling == "error"
                    else {
                        "output": list(content_parts(block.content)),
                    },
                }
            }

        elif isinstance(block, ModelReasoning):
            match block.meta.kind:
                case "thought":
                    if signature := block.meta.get_str("signature"):
                        yield {
                            "text": block.content.to_str(),
                            "thought": True,
                            "thought_signature": b64decode(signature),
                        }

                    else:
                        yield {
                            "text": block.content.to_str(),
                            "thought": True,
                        }

                case other:
                    raise ValueError(f"Unsupported reasoning element: {other}")

        else:
            yield from content_parts(block)


def content_parts(
    content: MultimodalContent,
    /,
) -> Generator[PartDict]:
    for part in content.parts:
        if isinstance(part, TextContent):
            yield {
                "text": part.text,
            }

        elif isinstance(part, ResourceContent):
            yield {
                "inline_data": {
                    # decode urlsafe base64 back to raw bytes for provider
                    "data": urlsafe_b64decode(part.data),
                    "mime_type": part.mime_type,
                },
            }

        elif isinstance(part, ResourceReference):
            yield {
                "file_data": {
                    "file_uri": part.uri,
                    "mime_type": part.mime_type,
                }
            }

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            # Skip artifacts that are marked as hidden
            if part.hidden:
                continue

            yield {
                "text": part.artifact.to_str(),
            }


def _normalized_schema(
    output: type[DataModel],
) -> SchemaDict:
    return cast(SchemaDict, _normalized_schema_content(output.__SPECIFICATION__))


def _normalized_schema_content(
    schema: Any,
) -> Any:
    if isinstance(schema, Mapping):
        return {
            key: _normalized_schema_content(value)
            for key, value in cast(Mapping[str, Any], schema).items()
            # it seems gemini is not supporting `additionalProperties` key within json schema
            if key != "additionalProperties"
        }

    if isinstance(schema, Sequence) and not isinstance(schema, str):
        return [_normalized_schema_content(element) for element in cast(Sequence[Any], schema)]

    return schema
