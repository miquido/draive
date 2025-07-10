import random
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from itertools import chain
from typing import Any, Literal, cast, overload

from google.api_core.exceptions import ResourceExhausted  # pyright: ignore[reportMissingImport]
from google.genai.types import (
    Candidate,
    ContentListUnionDict,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HarmCategory,
    MediaResolution,
    Modality,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import ObservabilityLevel, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiGenerationConfig, GeminiSafetyConfig
from draive.gemini.lmm import context_element_as_content, result_part_as_content_or_call
from draive.gemini.types import GeminiException
from draive.gemini.utils import unwrap_missing
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMInstruction,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamChunk,
    LMMStreamOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMTools,
)
from draive.lmm.helpers import lmm_output_decoder
from draive.lmm.types import LMMOutputDecoder
from draive.multimodal.content import MultimodalContent, MultimodalContentElement
from draive.utils import RateLimitError

__all__ = ("GeminiLMMGeneration",)


class GeminiLMMGeneration(GeminiAPI):
    def lmm(self) -> LMM:
        return LMM(completing=self.lmm_completion)

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        config: GeminiGenerationConfig | None = None,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: Literal[True],
        config: GeminiGenerationConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(
        self,
        *,
        instruction: LMMInstruction | None,
        context: LMMContext,
        tools: LMMTools | None,
        output: LMMOutputSelection,
        stream: bool = False,
        config: GeminiGenerationConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        """
        Generate completion using Gemini model.

        Args:
            instruction: System instruction for the model
            context: Context elements for the conversation
            tools: Available tools for the model to use
            output: Output specification
            stream: Whether to stream the response
            config: Generation configuration
            **extra: Additional configuration parameters

        Returns:
            LMM output or async iterator of stream outputs
        """
        tools = tools or LMMTools.none
        generation_config: GeminiGenerationConfig = config or ctx.state(GeminiGenerationConfig)

        with ctx.scope("gemini_lmm_completion", generation_config):
            request_config: GenerateContentConfigDict = _prepare_request_config(
                instruction=instruction,
                context=context,
                tools=tools,
                output=output,
                config=generation_config,
            )
            request_content: ContentListUnionDict = list(
                chain.from_iterable([context_element_as_content(element) for element in context])
            )
            output_decoder: LMMOutputDecoder = lmm_output_decoder(output)

            if stream:
                return await self._completion_stream(
                    model=generation_config.model,
                    content=request_content,
                    config=request_config,
                    output_decoder=output_decoder,
                )

            return await self._completion(
                model=generation_config.model,
                content=request_content,
                config=request_config,
                output_decoder=output_decoder,
            )

    async def _completion(  # noqa: C901, PLR0912
        self,
        model: str,
        content: ContentListUnionDict,
        config: GenerateContentConfigDict,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> LMMOutput:
        try:
            completion: GenerateContentResponse = await self._client.aio.models.generate_content(
                model=model,
                config=config,
                contents=content,
            )

        except ResourceExhausted as exc:
            ctx.record(
                ObservabilityLevel.WARNING,
                event="lmm.rate_limit",
            )
            raise RateLimitError(
                retry_after=random.uniform(0.3, 3.0),  # nosec: B311
            ) from exc

        except Exception as exc:
            raise GeminiException(f"Failed to generate Gemini completion: {exc}") from exc

        _record_usage_metrics(
            completion.usage_metadata,
            model=model,
        )

        if not completion.candidates:
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.completion.error",
                attributes={"error": "Missing candidates"},
            )
            raise GeminiException("Invalid Gemini completion - missing candidates!", completion)

        completion_candidate: Candidate = completion.candidates[0]  # we always request only one

        if completion_candidate.safety_ratings:
            ctx.record(
                ObservabilityLevel.INFO,
                event="lmm.safety.results",
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
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.safety.blocked",
                attributes={"reason": str(completion_candidate.finish_message)},
            )
            raise GeminiException("Gemini moderation blocked", completion)

        if not completion_candidate.content or not completion_candidate.content.parts:
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.completion.error",
                attributes={"error": "Missing content"},
            )
            raise GeminiException("Empty Gemini completion content!")

        result_content_elements: list[MultimodalContentElement] = []
        tool_requests: list[LMMToolRequest] = []
        for element in chain.from_iterable(
            result_part_as_content_or_call(part) for part in completion_candidate.content.parts
        ):
            if isinstance(element, LMMToolRequest):
                tool_requests.append(element)

            else:
                result_content_elements.append(element)

        completion_content: MultimodalContent
        if result_content_elements:
            completion_content = output_decoder(MultimodalContent.of(*result_content_elements))

        else:
            completion_content = MultimodalContent.empty

        if completion_candidate.finish_reason == FinishReason.MAX_TOKENS:
            ctx.record(
                ObservabilityLevel.WARNING,
                event="lmm.completion.warning",
                attributes={"finish_reason": str(completion_candidate.finish_message)},
            )

        elif completion_candidate.finish_reason != FinishReason.STOP:
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.completion.error",
                attributes={"finish_reason": str(completion_candidate.finish_message)},
            )
            raise GeminiException(f"Gemini completion error: {completion_candidate.finish_message}")

        if tool_requests:
            ctx.record(
                ObservabilityLevel.INFO,
                event="lmm.tool_requests",
                attributes={
                    "finish_reason": str(completion_candidate.finish_reason),
                    "lmm.tools": [call.tool for call in tool_requests],
                },
            )
            return LMMToolRequests.of(
                tool_requests,
                content=completion_content,
                meta={"finish_reason": str(completion_candidate.finish_reason)},
            )

        else:
            ctx.record(
                ObservabilityLevel.INFO,
                event="lmm.completion",
                attributes={"finish_reason": str(completion_candidate.finish_reason)},
            )
            return LMMCompletion.of(
                completion_content,
                meta={"finish_reason": str(completion_candidate.finish_reason)},
            )

    async def _completion_stream(
        self,
        model: str,
        content: ContentListUnionDict,
        config: GenerateContentConfigDict,
        output_decoder: LMMOutputDecoder,
    ) -> AsyncIterator[LMMStreamOutput]:
        completion_stream: AsyncIterator[GenerateContentResponse]
        try:
            completion_stream = await self._client.aio.models.generate_content_stream(
                model=model,
                config=config,
                contents=content,
            )

        except ResourceExhausted as exc:
            ctx.record(
                ObservabilityLevel.WARNING,
                event="lmm.rate_limit",
            )
            raise RateLimitError(
                retry_after=random.uniform(0.3, 3.0),  # nosec: B311
            ) from exc

        except Exception as exc:
            raise GeminiException(f"Failed to initialize Gemini streaming: {exc}") from exc

        return ctx.stream(
            _process_stream,
            stream=completion_stream,
            model=model,
            output_decoder=output_decoder,
        )


async def _process_stream(  # noqa PLR0912
    *,
    stream: AsyncIterator[GenerateContentResponse],
    model: str,
    output_decoder: LMMOutputDecoder,
) -> AsyncGenerator[LMMStreamChunk | LMMToolRequest]:
    async for chunk in stream:
        _record_usage_metrics(
            chunk.usage_metadata,
            model=model,
        )

        if not chunk.candidates:
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.completion.error",
                attributes={"error": "Missing candidates"},
            )
            raise GeminiException("Invalid Gemini chunk - missing candidates!", chunk)

        completion_candidate: Candidate = chunk.candidates[0]  # we always request only one

        if completion_candidate.safety_ratings:
            ctx.record(
                ObservabilityLevel.INFO,
                event="lmm.safety.results",
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
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.safety.blocked",
                attributes={"finish_reason": str(completion_candidate.finish_message)},
            )
            raise GeminiException("Gemini moderation blocked", chunk)

        if not completion_candidate.content or not completion_candidate.content.parts:
            match completion_candidate.finish_reason:
                case None:
                    ctx.record(
                        ObservabilityLevel.WARNING,
                        event="lmm.completion.warning",
                        attributes={"warning": "Empty chunk content"},
                    )
                    continue  # skip empty content

                case FinishReason.STOP:
                    ctx.record(
                        ObservabilityLevel.INFO,
                        event="lmm.completion",
                        attributes={"finish_reason": str(completion_candidate.finish_message)},
                    )
                    yield LMMStreamChunk.of(
                        MultimodalContent.empty,
                        eod=True,
                        meta={"finish_reason": str(completion_candidate.finish_message)},
                    )
                    return  # end of stream

                case FinishReason.MAX_TOKENS:
                    ctx.record(
                        ObservabilityLevel.WARNING,
                        event="lmm.completion.warning",
                        attributes={"finish_reason": str(completion_candidate.finish_message)},
                    )
                    yield LMMStreamChunk.of(
                        MultimodalContent.empty,
                        eod=True,
                        meta={"finish_reason": str(completion_candidate.finish_message)},
                    )
                    return  # end of stream

                case other:
                    ctx.record(
                        ObservabilityLevel.ERROR,
                        event="lmm.completion.error",
                        attributes={"finish_reason": str(other)},
                    )
                    raise GeminiException(f"Gemini completion error: {other}")

        chunk_content: list[MultimodalContentElement] = []
        for part in completion_candidate.content.parts:
            for element in result_part_as_content_or_call(part):
                if isinstance(element, LMMToolRequest):
                    ctx.record(
                        ObservabilityLevel.INFO,
                        event="lmm.tool_request",
                        attributes={"lmm.tool": element.tool},
                    )
                    yield element

                else:
                    chunk_content.append(element)

        if completion_candidate.finish_reason == FinishReason.STOP:
            yield LMMStreamChunk.of(
                output_decoder(MultimodalContent.of(*chunk_content)),
                eod=True,
                meta={"finish_reason": str(completion_candidate.finish_reason)},
            )
            return  # end of stream

        elif completion_candidate.finish_reason == FinishReason.MAX_TOKENS:
            ctx.record(
                ObservabilityLevel.WARNING,
                event="lmm.completion.warning",
                attributes={"finish_reason": str(completion_candidate.finish_message)},
            )

            yield LMMStreamChunk.of(
                output_decoder(MultimodalContent.of(*chunk_content)),
                eod=True,
                meta={"finish_reason": str(completion_candidate.finish_reason)},
            )
            return  # end of stream

        elif completion_candidate.finish_reason is not None:
            ctx.record(
                ObservabilityLevel.ERROR,
                event="lmm.completion.error",
                attributes={"finish_reason": str(completion_candidate.finish_message)},
            )
            raise GeminiException(f"Gemini completion error: {completion_candidate.finish_message}")

        else:  # continue with chunk
            yield LMMStreamChunk.of(
                MultimodalContent.of(*chunk_content),
                eod=False,
            )


def _record_usage_metrics(
    usage: GenerateContentResponseUsageMetadata | None,
    *,
    model: str,
) -> None:
    if usage is None:
        return

    ctx.record(
        ObservabilityLevel.INFO,
        metric="lmm.input_tokens",
        value=usage.prompt_token_count or 0,
        unit="tokens",
        attributes={"lmm.model": model},
    )
    ctx.record(
        ObservabilityLevel.INFO,
        metric="lmm.input_tokens.cached",
        value=usage.cached_content_token_count or 0,
        unit="tokens",
        attributes={"lmm.model": model},
    )
    ctx.record(
        ObservabilityLevel.INFO,
        metric="lmm.output_tokens",
        value=usage.candidates_token_count or 0,
        unit="tokens",
        attributes={"lmm.model": model},
    )


def _prepare_request_config(  # noqa: C901, PLR0912, PLR0915
    *,
    instruction: LMMInstruction | None,
    context: LMMContext,
    tools: LMMTools,
    output: LMMOutputSelection,
    config: GeminiGenerationConfig,
) -> GenerateContentConfigDict:
    temperature: float | None = unwrap_missing(config.temperature)
    max_tokens: int | None = unwrap_missing(config.max_tokens)
    thinking_budget: int | None = unwrap_missing(config.thinking_budget)
    seed: int | None = unwrap_missing(config.seed)
    ctx.record(
        ObservabilityLevel.INFO,
        attributes={
            "lmm.provider": "gemini",
            "lmm.model": config.model,
            "lmm.temperature": temperature,
            "lmm.max_tokens": max_tokens,
            "lmm.thinking_budget": thinking_budget,
            "lmm.instruction": instruction,
            "lmm.context": [element.to_str() for element in context],
            "lmm.tools": [tool["name"] for tool in tools.specifications],
            "lmm.tool_selection": f"{tools.selection}",
            "lmm.output": f"{output}",
            "lmm.seed": seed,
        },
    )

    response_schema: SchemaDict | None
    response_modalities: list[Modality] | None
    response_mime_type: str | None
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
            response_mime_type = None  # define mime type?

        case "audio":
            response_schema = None
            response_modalities = [Modality.AUDIO]
            response_mime_type = None

        case "video":
            raise NotImplementedError("video output is not supported by Gemini")

        case ["text", "image"] | ["image", "text"]:  # refine multimodal matching?
            response_schema = None
            response_modalities = [
                Modality.TEXT,
                Modality.IMAGE,
            ]
            response_mime_type = None

        case [*_]:
            raise NotImplementedError("multimodal output is not supported by Gemini")

        case model:
            response_schema = cast(SchemaDict, model.__PARAMETERS_SPECIFICATION__)
            response_modalities = [Modality.TEXT]
            response_mime_type = "application/json"

    functions: list[FunctionDeclarationDict] | None
    function_calling_mode: FunctionCallingConfigMode | None
    allowed_function_names: list[str] | None = None
    if tools.specifications:
        functions = [
            FunctionDeclarationDict(
                name=tool["name"],
                description=tool["description"],
                parameters=cast(SchemaDict, tool["parameters"]),
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
                allowed_function_names = [specific_tool["name"]]

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
        "max_output_tokens": max_tokens,
        "candidate_count": 1,
        "seed": seed,
        "stop_sequences": unwrap_missing(
            config.stop_sequences,
            transform=as_list,
        ),
        "safety_settings": [
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": safety_config.harm_category_hate_speech_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": safety_config.harm_category_dangerous_content_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": safety_config.harm_category_harassment_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": safety_config.harm_category_sexually_explicit_threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                "threshold": safety_config.harm_category_civic_integrity_threshold,
            },
        ],
        "system_instruction": instruction,
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
            "include_thoughts": True,
            "thinking_budget": thinking_budget,
        }
        if thinking_budget is not None
        else None,
    }
