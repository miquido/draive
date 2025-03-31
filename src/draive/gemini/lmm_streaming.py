from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from itertools import chain
from typing import Any

from google.genai.types import (
    Candidate,
    Content,
    ContentUnionDict,
    FinishReason,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import ArgumentsTrace, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiGenerationConfig
from draive.gemini.lmm import (
    DISABLED_SAFETY_SETTINGS,
    content_element_as_part,
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
    LMMContext,
    LMMStream,
    LMMStreamChunk,
    LMMStreamInput,
    LMMStreamOutput,
    LMMStreamProperties,
    LMMToolRequest,
)
from draive.lmm.types import LMMContextElement, LMMToolRequests, LMMToolResponse, LMMToolResponses
from draive.metrics import TokenUsage
from draive.multimodal import MultimodalContent
from draive.multimodal.content import MultimodalContentElement

__all__ = [
    "GeminiLMMStreaming",
]


class GeminiLMMStreaming(GeminiAPI):
    def lmm_streaming(self) -> LMMStream:
        async def prepare_stream(
            *,
            properties: AsyncIterator[LMMStreamProperties],
            input: AsyncIterator[LMMStreamInput],  # noqa: A002
            context: LMMContext | None,
            **extra: Any,
        ) -> AsyncIterator[LMMStreamOutput]:
            return self.prepare_lmm_stream(
                properties=properties,
                input=input,
                context=context,
                **extra,
            )

        return LMMStream(prepare=prepare_stream)

    async def prepare_lmm_stream(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        properties: AsyncIterator[LMMStreamProperties],
        input: AsyncIterator[LMMStreamInput],  # noqa: A002
        context: LMMContext | None,
        config: GeminiGenerationConfig | None = None,
        **extra: Any,
    ) -> AsyncGenerator[LMMStreamOutput]:
        generation_config: GeminiGenerationConfig = (
            config or ctx.state(GeminiGenerationConfig).updated()
        )
        ctx.record(
            ArgumentsTrace.of(
                config=generation_config,
                context=context,
                **extra,
            )
        )

        context_elements: Sequence[LMMContextElement]
        match context:
            case None:
                context_elements = ()

            case [*elements]:
                context_elements = elements

        content: list[ContentUnionDict] = list(
            chain.from_iterable(
                [context_element_as_content(element) for element in context_elements]
            )
        )

        # track requested tool calls out of the loop
        pending_tool_requests: list[LMMToolRequest] = []
        pending_tool_responses: list[LMMToolResponse] = []
        # before each call check for updated properties - this supposed to be an infinite loop
        async for current_properties in properties:
            # for each call accumulate input first
            input_buffer: MultimodalContent = MultimodalContent.empty
            async for chunk in input:
                match chunk:
                    # gether input content chunks until marked as end
                    case LMMStreamChunk() as content_chunk:
                        input_buffer = input_buffer.extending(content_chunk.content)
                        if content_chunk.eod:
                            content.append(
                                {
                                    "role": "user",
                                    "parts": [
                                        content_element_as_part(element)
                                        for element in input_buffer.parts
                                    ],
                                }
                            )
                            break  # we are supporting only completed input messages with this api

                    # accumulate tool results directly in context
                    case tool_result:
                        pending_tool_responses.append(tool_result)
                        # when there is no pending input and we got all requested tool results
                        if not input_buffer and len(pending_tool_requests) == len(
                            pending_tool_responses
                        ):
                            break  # then we can request completion again

            else:  # finalize streaming if input is finished
                return

            # add tool results to context
            if pending_tool_responses:
                content.extend(
                    context_element_as_content(
                        LMMToolResponses(
                            responses=pending_tool_responses,
                        ),
                    )
                )
                pending_tool_requests = []
                pending_tool_responses = []

            assert not pending_tool_requests  # nosec: B101

            # and prepare context for the next request
            response_schema: SchemaDict | None
            response_mime_type: str | None
            response_modalities: list[str] | None
            response_schema, response_modalities, response_mime_type, output_decoder = (
                output_as_response_declaration(current_properties.output)
            )

            functions: list[FunctionDeclarationDict] | None
            function_calling_mode: FunctionCallingConfigMode | None
            functions, function_calling_mode = tools_as_tools_config(
                current_properties.tools,
                tool_selection=current_properties.tool_selection,
            )

            accumulated_result: MultimodalContent = MultimodalContent.empty

            async for part in await self._client.aio.models.generate_content_stream(  # pyright: ignore[reportGeneralTypeIssues] - it has been marked incorrectly in google SDK
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
                    "system_instruction": Instruction.formatted(current_properties.instruction),
                    "tools": [{"function_declarations": functions}] if functions else None,
                    "tool_config": {"function_calling_config": {"mode": function_calling_mode}}
                    if function_calling_mode
                    else None,
                    # prevent google from automatically resolving function calls
                    "automatic_function_calling": {
                        "disable": True,
                        "maximum_remote_calls": None,
                    },
                    "media_resolution": resoluton_as_media_resulution(
                        generation_config.media_resolution
                    ),
                    "response_modalities": response_modalities,
                    "response_mime_type": response_mime_type,
                    "response_schema": response_schema,
                    "speech_config": unwrap_missing(
                        generation_config.speech_voice_name,
                        transform=_speech_config,
                    ),
                },
                contents=content,
            ):
                if usage := part.usage_metadata:
                    ctx.record(
                        TokenUsage.for_model(
                            generation_config.model,
                            input_tokens=usage.prompt_token_count,
                            cached_tokens=usage.cached_content_token_count,
                            output_tokens=usage.candidates_token_count,
                        ),
                    )

                if not part.candidates:
                    raise GeminiException(
                        "Invalid Gemini completion part - missing candidates!",
                        part,
                    )

                part_choice: Candidate = part.candidates[0]

                completion_part_content: Content
                if part_choice_content := part_choice.content:
                    completion_part_content = part_choice_content

                else:
                    raise GeminiException("Missing Gemini completion content!")

                part_content_elements: list[MultimodalContentElement] = []
                for element in chain.from_iterable(
                    result_part_as_content_or_call(part)
                    for part in completion_part_content.parts or []
                ):
                    match element:
                        case LMMToolRequest() as tool_request:
                            pending_tool_requests.append(tool_request)
                            yield tool_request

                        case content_element:
                            part_content_elements.append(content_element)

                if part_content_elements:
                    part_content: MultimodalContent = output_decoder(
                        MultimodalContent.of(*part_content_elements)
                    )
                    accumulated_result = accumulated_result.extending(part_content)
                    yield LMMStreamChunk.of(part_content)

                match part_choice.finish_reason:
                    case None:
                        continue

                    case FinishReason.STOP:
                        if pending_tool_requests:
                            content.extend(
                                context_element_as_content(
                                    LMMToolRequests(
                                        requests=pending_tool_requests,
                                    ),
                                )
                            )

                        else:
                            content.append(
                                {
                                    "role": "model",
                                    "parts": [
                                        content_element_as_part(element)
                                        for element in accumulated_result.parts
                                    ],
                                }
                            )
                            # send completion chunk - openAI sends it without an actual content
                            yield LMMStreamChunk.of(
                                MultimodalContent.empty,
                                eod=True,
                            )

                    case reason:  # other cases are errors
                        raise GeminiException(
                            f"Gemini completion generation failed! Reason: {reason}",
                            part,
                        )


def _speech_config(voice_name: str) -> SpeechConfigDict:
    return {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": voice_name},
        }
    }
