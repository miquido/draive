from collections.abc import Iterable
from itertools import chain
from typing import Any

from google.genai.types import (
    Candidate,
    Content,
    ContentListUnionDict,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    GenerateContentResponse,
    SchemaDict,
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
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMInvocation,
    LMMOutput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.metrics import TokenUsage
from draive.multimodal.content import MultimodalContent, MultimodalContentElement

__all__ = [
    "GeminiLMMInvoking",
]


class GeminiLMMInvoking(GeminiAPI):
    def lmm_invoking(self) -> LMMInvocation:
        return LMMInvocation(invoke=self.lmm_invocation)

    async def lmm_invocation(  # noqa: PLR0912
        self,
        *,
        instruction: Instruction | str | None,
        context: LMMContext,
        tool_selection: LMMToolSelection,
        tools: Iterable[LMMToolSpecification] | None,
        output: LMMOutputSelection,
        **extra: Any,
    ) -> LMMOutput:
        with ctx.scope("gemini_lmm_invocation"):
            ctx.record(
                ArgumentsTrace.of(
                    instruction=instruction,
                    context=context,
                    tools=tools,
                    tool_selection=tool_selection,
                    output=output,
                    **extra,
                )
            )
            config: GeminiGenerationConfig = ctx.state(GeminiGenerationConfig).updated(**extra)
            ctx.record(config)

            content: ContentListUnionDict = list(
                chain.from_iterable([context_element_as_content(element) for element in context])
            )

            response_schema: SchemaDict | None
            response_mime_type: str | None
            response_schema, response_mime_type, output_decoder = output_as_response_declaration(
                output
            )

            functions: list[FunctionDeclarationDict] | None
            function_calling_mode: FunctionCallingConfigMode | None
            functions, function_calling_mode = tools_as_tools_config(
                tools,
                tool_selection=tool_selection,
            )

            completion: GenerateContentResponse = await self._client.aio.models.generate_content(
                model=config.model,
                config={
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "max_output_tokens": config.max_tokens,
                    "candidate_count": 1,
                    "seed": config.seed,
                    "stop_sequences": as_list(config.stop_sequences)
                    if config.stop_sequences
                    else None,
                    # gemini safety is really bad and often triggers false positive
                    "safety_settings": DISABLED_SAFETY_SETTINGS,
                    "system_instruction": Instruction.formatted(instruction)
                    if instruction
                    else None,
                    "tools": [{"function_declarations": functions}] if functions else None,
                    "tool_config": {
                        "function_calling_config": {"mode": function_calling_mode},
                    }
                    if function_calling_mode
                    else None,
                    # prevent google from automatically resolving function calls
                    "automatic_function_calling": {"disable": True, "maximum_remote_calls": None},
                    "media_resolution": resoluton_as_media_resulution(config.media_resolution),
                    "response_schema": response_schema,
                    "response_mime_type": response_mime_type,
                },
                contents=content,
            )

            if usage := completion.usage_metadata:
                ctx.record(
                    TokenUsage.for_model(
                        config.model,
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
                    completion=lmm_completion,
                    requests=tool_requests,
                )
                ctx.record(ResultTrace.of(completion_tool_calls))
                return completion_tool_calls

            elif lmm_completion:
                ctx.record(ResultTrace.of(lmm_completion))
                return lmm_completion

            else:
                raise GeminiException("Invalid Gemini completion, missing content!", completion)
