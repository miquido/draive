from collections.abc import AsyncIterator, Iterable
from itertools import chain
from typing import Any, Literal, overload

from google.genai.types import (
    Candidate,
    Content,
    ContentListUnionDict,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentResponse,
    Modality,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import ArgumentsTrace, ResultTrace, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiGenerationConfig
from draive.gemini.lmm import (
    DISABLED_SAFETY_SETTINGS,
    context_element_as_content,
    output_as_response_declaration,
    resoluton_as_media_resulution,
    result_part_as_content_or_call,
    tools_as_tools_config,
)
from draive.gemini.types import GeminiException
from draive.gemini.utils import unwrap_missing
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamOutput,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.metrics import TokenUsage
from draive.multimodal.content import MultimodalContent, MultimodalContentElement

__all__ = ("GeminiLMMGeneration",)


class GeminiLMMGeneration(GeminiAPI):
    def lmm(self) -> LMM:
        return LMM(completing=self.lmm_completion)

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        config: GeminiGenerationConfig | None = None,
        output: LMMOutputSelection,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def lmm_completion(
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        config: GeminiGenerationConfig | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    async def lmm_completion(  # noqa: C901, PLR0912
        self,
        *,
        instruction: Instruction | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        config: GeminiGenerationConfig | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            raise NotImplementedError("gemini streaming is not implemented yet")

        with ctx.scope("gemini_lmm_completion"):
            generation_config: GeminiGenerationConfig = config or ctx.state(GeminiGenerationConfig)
            ctx.record(
                ArgumentsTrace.of(
                    config=generation_config,
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                )
            )

            content: ContentListUnionDict = list(
                chain.from_iterable([context_element_as_content(element) for element in context])
            )

            response_schema: SchemaDict | None
            response_mime_type: str | None
            response_modalities: list[Modality] | None
            response_schema, response_modalities, response_mime_type, output_decoder = (
                output_as_response_declaration(output)
            )

            functions: list[FunctionDeclarationDict] | None
            function_calling_mode: FunctionCallingConfigMode | None
            functions, function_calling_mode = tools_as_tools_config(
                tools,
                tool_selection=tool_selection,
            )

            completion: GenerateContentResponse = await self._client.aio.models.generate_content(
                model=generation_config.model,
                config={
                    "temperature": unwrap_missing(generation_config.temperature),
                    "top_p": unwrap_missing(generation_config.top_p),
                    "top_k": unwrap_missing(generation_config.top_k),
                    "max_output_tokens": unwrap_missing(generation_config.max_tokens),
                    "candidate_count": 1,
                    "seed": unwrap_missing(generation_config.seed),
                    "stop_sequences": unwrap_missing(
                        generation_config.stop_sequences,
                        transform=as_list,
                    ),
                    # gemini safety is really bad and often triggers false positive
                    "safety_settings": DISABLED_SAFETY_SETTINGS,
                    "system_instruction": Instruction.formatted(instruction),
                    "tools": [{"function_declarations": functions}] if functions else None,
                    "tool_config": {"function_calling_config": {"mode": function_calling_mode}}
                    if function_calling_mode
                    else None,
                    # prevent google from automatically resolving function calls
                    "automatic_function_calling": {"disable": True, "maximum_remote_calls": None},
                    "media_resolution": resoluton_as_media_resulution(
                        generation_config.media_resolution
                    ),
                    "response_modalities": [str(modality) for modality in response_modalities]
                    if response_modalities is not None
                    else None,
                    "response_mime_type": response_mime_type,
                    "response_schema": response_schema,
                    "speech_config": unwrap_missing(
                        generation_config.speech_voice_name,
                        transform=_speech_config,
                    ),
                },
                contents=content,
            )

            if usage := completion.usage_metadata:
                ctx.record(
                    TokenUsage.for_model(
                        generation_config.model,
                        input_tokens=usage.prompt_token_count,
                        cached_tokens=usage.cached_content_token_count,
                        output_tokens=usage.candidates_token_count,
                    ),
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
                assert tools, "Requesting tool call without tools"  # nosec: B101
                completion_tool_calls = LMMToolRequests(
                    content=lmm_completion.content if lmm_completion else None,
                    requests=tool_requests,
                )
                ctx.record(ResultTrace.of(completion_tool_calls))
                return completion_tool_calls

            elif lmm_completion:
                ctx.record(ResultTrace.of(lmm_completion))
                return lmm_completion

            else:
                raise GeminiException("Invalid Gemini completion, missing content!", completion)


def _speech_config(voice_name: str) -> SpeechConfigDict:
    return {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": voice_name},
        }
    }
