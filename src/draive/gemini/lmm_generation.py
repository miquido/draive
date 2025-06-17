from collections.abc import AsyncIterator, Callable
from itertools import chain
from typing import Any, Literal, overload

from google.api_core.exceptions import ResourceExhausted
from google.genai.types import (
    Candidate,
    Content,
    ContentListUnionDict,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentResponse,
    MediaResolution,
    Modality,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import ObservabilityLevel, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiGenerationConfig
from draive.gemini.lmm import (
    DISABLED_SAFETY_SETTINGS,
    context_element_as_content,
    output_as_response_declaration,
    resolution_as_media_resolution,
    result_part_as_content_or_call,
    tools_as_tools_config,
)
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
        tools = tools or LMMTools.none
        generation_config: GeminiGenerationConfig = config or ctx.state(
            GeminiGenerationConfig
        ).updated(**extra)
        with ctx.scope("gemini_lmm_completion", generation_config):
            ctx.record(
                ObservabilityLevel.INFO,
                attributes={
                    "lmm.provider": "gemini",
                    "lmm.model": generation_config.model,
                    "lmm.temperature": generation_config.temperature,
                    "lmm.max_tokens": generation_config.max_tokens,
                    "lmm.seed": generation_config.seed,
                    "lmm.tools": [tool["name"] for tool in tools.specifications],
                    "lmm.tool_selection": f"{tools.selection}",
                    "lmm.stream": stream,
                    "lmm.output": f"{output}",
                    "lmm.instruction": f"{instruction}",
                    "lmm.context": [element.to_str() for element in context],
                },
            )

            content: ContentListUnionDict = list(
                chain.from_iterable([context_element_as_content(element) for element in context])
            )

            response_schema: SchemaDict | None
            response_modalities: list[Modality] | None
            response_mime_type: str | None
            response_schema, response_modalities, response_mime_type, output_decoder = (
                output_as_response_declaration(output)
            )

            functions: list[FunctionDeclarationDict] | None
            function_calling_mode: FunctionCallingConfigMode | None
            functions, function_calling_mode = tools_as_tools_config(
                tools.specifications,
                tool_selection=tools.selection,
            )

            if stream:
                return await self._completion_stream(
                    model=generation_config.model,
                    temperature=unwrap_missing(generation_config.temperature),
                    top_p=unwrap_missing(generation_config.top_p),
                    top_k=unwrap_missing(generation_config.top_k),
                    max_tokens=unwrap_missing(generation_config.max_tokens),
                    seed=unwrap_missing(generation_config.seed),
                    stop_sequences=unwrap_missing(
                        generation_config.stop_sequences,
                        transform=as_list,
                    ),
                    instruction=instruction,
                    content=content,
                    functions=functions,
                    function_calling_mode=function_calling_mode,
                    media_resolution=resolution_as_media_resolution(
                        generation_config.media_resolution
                    ),
                    response_modalities=[m.name for m in response_modalities]
                    if response_modalities
                    else None,
                    response_mime_type=response_mime_type,
                    response_schema=response_schema,
                    speech_voice_name=unwrap_missing(
                        generation_config.speech_voice_name,
                        transform=_speech_config,
                    ),
                    output_decoder=output_decoder,
                )

            else:
                return await self._completion(
                    model=generation_config.model,
                    temperature=unwrap_missing(generation_config.temperature),
                    top_p=unwrap_missing(generation_config.top_p),
                    top_k=unwrap_missing(generation_config.top_k),
                    max_tokens=unwrap_missing(generation_config.max_tokens),
                    seed=unwrap_missing(generation_config.seed),
                    stop_sequences=unwrap_missing(
                        generation_config.stop_sequences,
                        transform=as_list,
                    ),
                    instruction=instruction,
                    content=content,
                    functions=functions,
                    function_calling_mode=function_calling_mode,
                    media_resolution=resolution_as_media_resolution(
                        generation_config.media_resolution
                    ),
                    response_modalities=[m.name for m in response_modalities]
                    if response_modalities
                    else None,
                    response_mime_type=response_mime_type,
                    response_schema=response_schema,
                    speech_voice_name=unwrap_missing(
                        generation_config.speech_voice_name,
                        transform=_speech_config,
                    ),
                    output_decoder=output_decoder,
                )

    async def _completion(  # noqa: C901, PLR0912, PLR0913
        self,
        model: str,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int | None,
        seed: int | None,
        stop_sequences: list[str] | None,
        instruction: str | None,
        content: ContentListUnionDict,
        functions: list[FunctionDeclarationDict] | None,
        function_calling_mode: FunctionCallingConfigMode | None,
        media_resolution: MediaResolution | None,
        response_modalities: list[str] | None,
        response_mime_type: str | None,
        response_schema: SchemaDict | None,
        speech_voice_name: SpeechConfigDict | None,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> LMMOutput:
        try:
            completion: GenerateContentResponse = await self._client.aio.models.generate_content(
                model=model,
                config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_output_tokens": max_tokens,
                    "candidate_count": 1,
                    "seed": seed,
                    "stop_sequences": stop_sequences,
                    # gemini safety is really bad and often triggers false positive
                    "safety_settings": DISABLED_SAFETY_SETTINGS,
                    "system_instruction": instruction if instruction is not None else {},
                    "tools": [{"function_declarations": functions}] if functions else None,
                    "tool_config": {"function_calling_config": {"mode": function_calling_mode}}
                    if function_calling_mode
                    else None,
                    # prevent google from automatically resolving function calls
                    "automatic_function_calling": {
                        "disable": True,
                        "maximum_remote_calls": None,
                    },
                    "media_resolution": media_resolution,
                    "response_modalities": response_modalities,
                    "response_mime_type": response_mime_type,
                    "response_schema": response_schema,
                    "speech_config": speech_voice_name,
                },
                contents=content,
            )
        except ResourceExhausted as exc:  # retry on rate limit after delay
            ctx.record(
                ObservabilityLevel.WARNING,
                event="lmm.rate_limit",
            )
            # Google's ResourceExhausted doesn't have retry-after header like others
            # so we'll let the RateLimitError default behavior handle it
            raise RateLimitError(retry_after=0) from exc

        except Exception as exc:
            raise GeminiException(f"Failed to generate Gemini completion: {exc}") from exc

        if usage := completion.usage_metadata:
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

        if not completion.candidates:
            raise GeminiException(
                "Invalid Gemini completion - missing candidates!",
                completion,
            )

        completion_choice: Candidate = completion.candidates[0]

        match completion_choice.finish_reason:
            case FinishReason.STOP:
                pass  # process results

            case FinishReason.MAX_TOKENS:
                raise GeminiException(
                    "Invalid Gemini completion - exceeded maximum length!",
                    completion,
                )

            case reason:  # other cases are errors
                raise GeminiException(
                    f"Gemini completion generation failed! Reason: {reason}",
                    completion,
                )

        completion_content: Content
        if candidate_content := completion_choice.content:
            completion_content = candidate_content

        else:
            raise GeminiException("Missing Gemini completion content!")

        result_content_elements: list[MultimodalContentElement] = []
        tool_requests: list[LMMToolRequest] = []
        for element in chain.from_iterable(
            result_part_as_content_or_call(part) for part in completion_content.parts or []
        ):
            match element:
                case LMMToolRequest() as tool_request:
                    tool_requests.append(tool_request)

                case content_element:
                    result_content_elements.append(content_element)

        lmm_completion: LMMCompletion | None
        if result_content_elements:
            lmm_completion = LMMCompletion.of(
                output_decoder(MultimodalContent.of(*result_content_elements))
            )

        else:
            lmm_completion = None

        if tool_requests:
            if not functions:
                raise GeminiException(
                    "Received tool call requests but no tools were provided in the configuration. "
                    "This indicates a mismatch between the model's response and the provided tools."
                )
            completion_tool_calls = LMMToolRequests(
                content=lmm_completion.content if lmm_completion else None,
                requests=tool_requests,
            )

            ctx.record(
                ObservabilityLevel.INFO,
                event="lmm.tool_requests",
                attributes={"lmm.tools": [call.tool for call in tool_requests]},
            )

            return completion_tool_calls

        elif lmm_completion:
            ctx.record(
                ObservabilityLevel.INFO,
                event="lmm.completion",
            )
            return lmm_completion

        else:
            raise GeminiException("Invalid Gemini completion, missing content!", completion)

    async def _completion_stream(  # noqa: C901, PLR0913
        self,
        model: str,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        max_tokens: int | None,
        seed: int | None,
        stop_sequences: list[str] | None,
        instruction: str | None,
        content: ContentListUnionDict,
        functions: list[FunctionDeclarationDict] | None,
        function_calling_mode: FunctionCallingConfigMode | None,
        media_resolution: MediaResolution | None,
        response_modalities: list[str] | None,
        response_mime_type: str | None,
        response_schema: SchemaDict | None,
        speech_voice_name: SpeechConfigDict | None,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> AsyncIterator[LMMStreamOutput]:
        # Create the streaming request with proper error handling
        try:
            completion_stream = await self._client.aio.models.generate_content_stream(
                model=model,
                config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_output_tokens": max_tokens,
                    "candidate_count": 1,
                    "seed": seed,
                    "stop_sequences": stop_sequences,
                    # gemini safety is really bad and often triggers false positive
                    "safety_settings": DISABLED_SAFETY_SETTINGS,
                    "system_instruction": instruction if instruction is not None else {},
                    "tools": [{"function_declarations": functions}] if functions else None,
                    "tool_config": {"function_calling_config": {"mode": function_calling_mode}}
                    if function_calling_mode
                    else None,
                    # prevent google from automatically resolving function calls
                    "automatic_function_calling": {"disable": True, "maximum_remote_calls": None},
                    "media_resolution": media_resolution,
                    "response_modalities": response_modalities,
                    "response_mime_type": response_mime_type,
                    "response_schema": response_schema,
                    "speech_config": speech_voice_name,
                },
                contents=content,
            )
        except Exception as exc:
            raise GeminiException(f"Failed to initialize Gemini streaming: {exc}") from exc

        async def stream():  # noqa: C901, PLR0912
            accumulated_tool_calls: list[LMMToolRequest] = []

            async for completion_chunk in completion_stream:
                # Record usage if available (expected in the last chunk)
                if usage := completion_chunk.usage_metadata:
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

                if not completion_chunk.candidates:
                    continue  # Skip chunks without candidates

                completion_choice: Candidate = completion_chunk.candidates[0]

                # Handle the content of the chunk
                if completion_choice.content and completion_choice.content.parts:
                    for part in completion_choice.content.parts:
                        # Process content and tool calls from each part
                        for element in result_part_as_content_or_call(part):
                            match element:
                                case LMMToolRequest() as tool_request:
                                    # Accumulate tool calls (they may come in multiple chunks)
                                    # Find existing tool call with the same identifier
                                    existing_tool_call = next(
                                        (
                                            call
                                            for call in accumulated_tool_calls
                                            if call.identifier == tool_request.identifier
                                        ),
                                        None,
                                    )

                                    if existing_tool_call:
                                        # Merge arguments for existing tool call
                                        merged_arguments: Any
                                        if isinstance(tool_request.arguments, dict) and isinstance(
                                            existing_tool_call.arguments, dict
                                        ):
                                            # Merge dictionaries for partial function call arguments
                                            merged_arguments = {
                                                **existing_tool_call.arguments,
                                                **tool_request.arguments,
                                            }
                                        else:
                                            # Non-dict arguments: replace entirely
                                            merged_arguments = tool_request.arguments

                                        # Replace the existing tool call with updated version
                                        for i, call in enumerate(accumulated_tool_calls):
                                            if call.identifier == tool_request.identifier:
                                                accumulated_tool_calls[i] = LMMToolRequest(
                                                    identifier=existing_tool_call.identifier,
                                                    tool=existing_tool_call.tool,
                                                    arguments=merged_arguments,
                                                    meta=existing_tool_call.meta,
                                                )
                                                break
                                    else:
                                        # Add new tool call
                                        accumulated_tool_calls.append(tool_request)

                                case content_element:
                                    # Yield content chunks immediately
                                    yield LMMStreamChunk.of(
                                        output_decoder(MultimodalContent.of(content_element))
                                    )

                # Handle finish reason
                if finish_reason := completion_choice.finish_reason:
                    match finish_reason:
                        case FinishReason.STOP:
                            # Send accumulated tool calls if any
                            if accumulated_tool_calls:
                                if not functions:
                                    raise GeminiException(
                                        "Received tool call requests but no tools were \
                                        provided in the configuration. "
                                        "This indicates a mismatch between the model's \
                                        response and the provided tools."
                                    )

                                for tool_request in accumulated_tool_calls:
                                    ctx.record(
                                        ObservabilityLevel.INFO,
                                        event="lmm.tool_request",
                                        attributes={"lmm.tool": tool_request.tool},
                                    )
                                    yield tool_request
                            else:
                                # Send completion marker if no tool calls
                                yield LMMStreamChunk.of(
                                    MultimodalContent.empty,
                                    eod=True,
                                )
                            break

                        case FinishReason.MAX_TOKENS:
                            raise GeminiException(
                                "Invalid Gemini completion - exceeded maximum length!",
                                completion_chunk,
                            )

                        case reason:  # other cases are errors
                            raise GeminiException(
                                f"Gemini completion generation failed! Reason: {reason}",
                                completion_chunk,
                            )

        return ctx.stream(stream)


def _speech_config(voice_name: str) -> SpeechConfigDict:
    return {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": voice_name},
        }
    }
