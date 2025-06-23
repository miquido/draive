import random
from collections.abc import AsyncIterator, Callable
from itertools import chain
from typing import Any, Literal, overload

from google.api_core.exceptions import ResourceExhausted  # pyright: ignore[reportMissingImport]
from google.genai.types import (
    Candidate,
    Content,
    ContentListUnionDict,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
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

            request_config = _build_request(
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
                functions=functions,
                function_calling_mode=function_calling_mode,
                media_resolution=resolution_as_media_resolution(generation_config.media_resolution),
                response_modalities=[m.name for m in response_modalities]
                if response_modalities
                else None,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                speech_voice_name=unwrap_missing(
                    generation_config.speech_voice_name,
                    transform=_speech_config,
                ),
            )

            if stream:
                return await self._completion_stream(
                    model=generation_config.model,
                    content=content,
                    config=request_config,
                    functions=functions,
                    output_decoder=output_decoder,
                )
            return await self._completion(
                model=generation_config.model,
                content=content,
                config=request_config,
                functions=functions,
                output_decoder=output_decoder,
            )

    async def _completion(
        self,
        model: str,
        content: ContentListUnionDict,
        config: GenerateContentConfigDict,
        functions: list[FunctionDeclarationDict] | None,
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

        self._record_usage_metrics(
            completion.usage_metadata,
            model=model,
        )

        if not completion.candidates:
            raise GeminiException("Invalid Gemini completion - missing candidates!", completion)

        completion_choice: Candidate = completion.candidates[0]
        self._validate_finish_reason(
            completion_choice.finish_reason,
            completion=completion,
        )

        completion_content: Content = self._extract_completion_content(completion_choice)
        result_content_elements: list[MultimodalContentElement]
        tool_requests: list[LMMToolRequest]
        result_content_elements, tool_requests = self._process_completion_parts(completion_content)

        lmm_completion: LMMCompletion | None = self._create_lmm_completion(
            result_content_elements,
            output_decoder=output_decoder,
        )

        return self._handle_completion_result(
            lmm_completion,
            tool_requests=tool_requests,
            functions=functions,
        )

    async def _completion_stream(
        self,
        model: str,
        content: ContentListUnionDict,
        config: GenerateContentConfigDict,
        functions: list[FunctionDeclarationDict] | None,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> AsyncIterator[LMMStreamOutput]:
        """Generate streaming completion."""
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

        async def stream():
            async for chunk in self._stream_processor(
                completion_stream, model, functions, output_decoder
            ):
                yield chunk

        return ctx.stream(stream)

    async def _stream_processor(  # noqa PLR0912
        self,
        completion_stream,
        model: str,
        functions: list[FunctionDeclarationDict] | None,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ):
        accumulated_tool_calls: list[LMMToolRequest] = []

        async for completion_chunk in completion_stream:
            # Record usage if available (expected in the last chunk)
            self._record_usage_metrics(
                completion_chunk.usage_metadata,
                model=model,
            )

            if not completion_chunk.candidates:
                continue  # Skip chunks without candidates

            completion_choice: Candidate = completion_chunk.candidates[0]

            # Handle the content of the chunk
            if completion_choice.content and completion_choice.content.parts:
                for part in completion_choice.content.parts:
                    for element in result_part_as_content_or_call(part):
                        if isinstance(element, LMMToolRequest):
                            accumulated_tool_calls = self._accumulate_tool_call(
                                accumulated_tool_calls, element
                            )

                        else:
                            yield LMMStreamChunk.of(
                                output_decoder(
                                    MultimodalContent.of(element),
                                ),
                            )

            # Handle finish reason
            if finish_reason := completion_choice.finish_reason:
                if finish_reason == FinishReason.STOP:
                    if accumulated_tool_calls:
                        if not functions:
                            raise GeminiException(
                                "Received tool call requests but no tools were provided "
                                "in the configuration. This indicates a mismatch between "
                                "the model's response and the provided tools."
                            )

                        for tool_request in accumulated_tool_calls:
                            ctx.record(
                                ObservabilityLevel.INFO,
                                event="lmm.tool_request",
                                attributes={"lmm.tool": tool_request.tool},
                            )
                            yield tool_request

                    else:
                        yield LMMStreamChunk.of(
                            MultimodalContent.empty,
                            eod=True,
                        )

                    return  # end of processing

                elif finish_reason == FinishReason.MAX_TOKENS:
                    raise GeminiException(
                        "Invalid Gemini completion - exceeded maximum length!",
                        completion_chunk,
                    )

                else:
                    raise GeminiException(
                        f"Gemini completion generation failed! Reason: {finish_reason}",
                        completion_chunk,
                    )

    def _record_usage_metrics(
        self,
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

    def _validate_finish_reason(
        self,
        finish_reason: FinishReason | None,
        *,
        completion: GenerateContentResponse,
    ) -> None:
        match finish_reason:
            case None:
                pass  # not finished

            case FinishReason.STOP:
                pass  # Valid completion

            case FinishReason.MAX_TOKENS:
                raise GeminiException(
                    "Invalid Gemini completion - exceeded maximum length!",
                    completion,
                )

            case reason:
                raise GeminiException(
                    f"Gemini completion generation failed! Reason: {reason}",
                    completion,
                )

    def _extract_completion_content(
        self,
        completion_choice: Candidate,
    ) -> Content:
        if candidate_content := completion_choice.content:
            return candidate_content

        else:
            raise GeminiException("Missing Gemini completion content!")

    def _process_completion_parts(
        self,
        completion_content: Content,
    ) -> tuple[list[MultimodalContentElement], list[LMMToolRequest]]:
        result_content_elements: list[MultimodalContentElement] = []
        tool_requests: list[LMMToolRequest] = []

        for element in chain.from_iterable(
            result_part_as_content_or_call(part) for part in completion_content.parts or []
        ):
            if isinstance(element, LMMToolRequest):
                tool_requests.append(element)
            else:
                result_content_elements.append(element)

        return result_content_elements, tool_requests

    def _create_lmm_completion(
        self,
        result_content_elements: list[MultimodalContentElement],
        *,
        output_decoder: Callable[[MultimodalContent], MultimodalContent],
    ) -> LMMCompletion | None:
        """Create LMM completion from content elements."""
        if result_content_elements:
            return LMMCompletion.of(output_decoder(MultimodalContent.of(*result_content_elements)))
        return None

    def _handle_completion_result(
        self,
        lmm_completion: LMMCompletion | None,
        *,
        tool_requests: list[LMMToolRequest],
        functions: list[FunctionDeclarationDict] | None,
    ) -> LMMOutput:
        """Handle the final completion result."""
        if tool_requests:
            if not functions:
                raise GeminiException(
                    "Received tool call requests but no tools were provided "
                    "in the configuration. This indicates a mismatch between "
                    "the model's response and the provided tools."
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
            raise GeminiException("Invalid Gemini completion, missing content!")

    def _accumulate_tool_call(
        self, accumulated_tool_calls: list[LMMToolRequest], tool_request: LMMToolRequest
    ) -> list[LMMToolRequest]:
        """Accumulate tool calls, merging partial calls with the same identifier."""
        existing_tool_call = next(
            (call for call in accumulated_tool_calls if call.identifier == tool_request.identifier),
            None,
        )

        if existing_tool_call:
            # Merge arguments for existing tool call
            merged_arguments: Any
            if isinstance(tool_request.arguments, dict) and isinstance(
                existing_tool_call.arguments, dict
            ):
                merged_arguments = {
                    **existing_tool_call.arguments,
                    **tool_request.arguments,
                }
            else:
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
            accumulated_tool_calls.append(tool_request)

        return accumulated_tool_calls


def _build_request(  # noqa PLR0913
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int | None,
    seed: int | None,
    stop_sequences: list[str] | None,
    instruction: str | None,
    functions: list[FunctionDeclarationDict] | None,
    function_calling_mode: FunctionCallingConfigMode | None,
    media_resolution: MediaResolution | None,
    response_modalities: list[str] | None,
    response_mime_type: str | None,
    response_schema: SchemaDict | None,
    speech_voice_name: SpeechConfigDict | None,
) -> GenerateContentConfigDict:
    """Build request configuration for Gemini API."""
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_tokens,
        "candidate_count": 1,
        "seed": seed,
        "stop_sequences": stop_sequences,
        # gemini safety is really bad and often triggers false positive
        "safety_settings": DISABLED_SAFETY_SETTINGS,
        "system_instruction": instruction,
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
    }


def _speech_config(voice_name: str) -> SpeechConfigDict:
    return {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": voice_name},
        }
    }
