# LMMSession
from asyncio import Task, TaskGroup
from collections.abc import AsyncIterator, Sequence
from itertools import chain
from typing import Any

from google.genai.live import AsyncSession
from google.genai.types import (
    Content,
    ContentDict,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    LiveConnectConfigDict,
    Modality,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import ArgumentsTrace, AsyncQueue, as_list, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiLiveConfig
from draive.gemini.lmm import (
    content_element_as_part,
    context_element_as_content,
    output_as_response_declaration,
    result_part_as_content_or_call,
    tools_as_tools_config,
)
from draive.gemini.utils import unwrap_missing
from draive.instructions import Instruction
from draive.lmm import (
    LMMContext,
    LMMSession,
    LMMSessionOutputSelection,
    LMMStreamInput,
    LMMStreamOutput,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.lmm.types import LMMStreamChunk, LMMToolRequest, LMMToolResponse
from draive.multimodal.content import MultimodalContent
from draive.multimodal.media import MediaContent
from draive.multimodal.text import TextContent
from draive.parameters.model import DataModel

__all__ = [
    "GeminiLMMSession",
]


class GeminiLMMSession(GeminiAPI):
    def lmm_session(self) -> LMMSession:
        return LMMSession(preparing=self.lmm_session_prepare)

    async def lmm_session_prepare(
        self,
        *,
        instruction: Instruction | None = None,
        initial_context: LMMContext | None = None,
        input_stream: AsyncIterator[LMMStreamInput],
        output: LMMSessionOutputSelection = "auto",
        tools: Sequence[LMMToolSpecification] | None = None,
        tool_selection: LMMToolSelection = "auto",
        config: GeminiLiveConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]:
        with ctx.scope("gemini_live"):
            live_config: GeminiLiveConfig = config or ctx.state(GeminiLiveConfig)
            ctx.record(
                ArgumentsTrace.of(
                    config=live_config,
                    instruction=instruction,
                    initial_context=initial_context,
                    output=output,
                    tools=tools,
                    tool_selection=tool_selection,
                )
            )
            context_elements: LMMContext
            match initial_context:
                case None:
                    context_elements = ()

                case [*elements]:
                    context_elements = elements

            initial_content: list[Content | ContentDict] = list(
                chain.from_iterable(
                    [context_element_as_content(element) for element in context_elements]
                )
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

            output_queue = AsyncQueue[LMMStreamOutput]()

            ctx.spawn(
                self._drive_connection,
                model=live_config.model,
                config={
                    "system_instruction": {"parts": [{"text": Instruction.formatted(instruction)}]},
                    "temperature": unwrap_missing(live_config.temperature),
                    "top_p": unwrap_missing(live_config.top_p),
                    "top_k": unwrap_missing(live_config.top_k),
                    "max_output_tokens": unwrap_missing(live_config.max_tokens),
                    "tools": [{"function_declarations": functions}] if functions else None,
                    "response_modalities": response_modalities
                    if response_modalities is not None
                    else None,
                    "seed": unwrap_missing(live_config.seed),
                    "generation_config": {
                        "temperature": unwrap_missing(live_config.temperature),
                        "top_p": unwrap_missing(live_config.top_p),
                        "top_k": unwrap_missing(live_config.top_k),
                        "max_output_tokens": unwrap_missing(live_config.max_tokens),
                        "candidate_count": 1,
                        "seed": unwrap_missing(live_config.seed),
                        "stop_sequences": unwrap_missing(
                            live_config.stop_sequences,
                            transform=as_list,
                        ),
                        "response_mime_type": response_mime_type,
                        "response_schema": response_schema,
                    },
                    "speech_config": unwrap_missing(
                        live_config.speech_voice_name,
                        transform=_speech_config,
                    ),
                },
                input_content=initial_content,
                input_stream=input_stream,
                output_stream=output_queue,
            )

            return output_queue

    @ctx.scope("connection", task_group=TaskGroup())
    async def _drive_connection(  # noqa: C901, PLR0912
        self,
        model: str,
        config: LiveConnectConfigDict,
        input_content: list[Content | ContentDict],
        input_stream: AsyncIterator[LMMStreamInput],
        output_stream: AsyncQueue[LMMStreamOutput],
    ):
        try:
            async with self._client.aio.live.connect(
                model=model,
                config=config,
            ) as session:
                output_task: Task[None] = ctx.spawn(
                    self._handle_output,
                    session=session,
                    output_stream=output_stream,
                )
                try:
                    if input_content:
                        await session.send_client_content(
                            turns=input_content,
                            turn_complete=True,
                        )
                    async for element in input_stream:
                        match element:
                            case LMMStreamChunk() as chunk:
                                for part in chunk.content.parts:
                                    match part:
                                        case TextContent() as text:
                                            await session.send_realtime_input(
                                                media={
                                                    "data": text.text.encode(),
                                                    "mime_type": "text/plain",
                                                },
                                            )

                                        case MediaContent() as media:
                                            match media.source:
                                                case str():
                                                    raise ValueError(
                                                        "Media reference is not supported"
                                                    )

                                                case bytes() as data:
                                                    await session.send_realtime_input(
                                                        media={
                                                            "data": data,
                                                            "mime_type": media.media,
                                                        },
                                                    )

                                        case DataModel() as data:
                                            await session.send_realtime_input(
                                                media={
                                                    "data": data.as_json().encode(),
                                                    "mime_type": "application/json",
                                                },
                                            )

                            case LMMToolResponse() as tool_response:
                                await session.send_tool_response(
                                    function_responses={
                                        "id": tool_response.identifier,
                                        "name": tool_response.tool,
                                        "response": {
                                            "error": [
                                                content_element_as_part(element)
                                                for element in tool_response.content.parts
                                            ],
                                        }
                                        if tool_response.error
                                        else {
                                            "output": [
                                                content_element_as_part(element)
                                                for element in tool_response.content.parts
                                            ],
                                        },
                                    }
                                )
                finally:
                    # cancel output on end
                    output_task.cancel()

        except BaseException as exc:
            output_stream.finish(exc)

        else:
            output_stream.finish()

    async def _handle_output(
        self,
        session: AsyncSession,
        output_stream: AsyncQueue[LMMStreamOutput],
    ) -> None:
        while not output_stream.is_finished:
            async for element in session.receive():
                if tool_calls := element.tool_call:
                    for call in tool_calls.function_calls or ():
                        if call.name is None or call.id is None:
                            continue  # skip unknown or incomplete tools

                        output_stream.enqueue(
                            LMMToolRequest(
                                identifier=call.id,
                                tool=call.name,
                                arguments=call.args or {},
                            )
                        )
                if content := element.server_content:
                    eod: bool = (
                        (content.turn_complete or False)
                        or (content.interrupted or False)
                        or (content.generation_complete or False)
                    )
                    if generation := content.model_turn:
                        output_stream.enqueue(
                            LMMStreamChunk.of(
                                MultimodalContent.of(
                                    *chain.from_iterable(
                                        result_part_as_content_or_call(part)
                                        for part in generation.parts or []
                                    )
                                ),
                                meta={
                                    "interrupted": content.interrupted,
                                },
                                eod=eod,
                            )
                        )

                    elif eod:
                        output_stream.enqueue(
                            LMMStreamChunk.of(
                                MultimodalContent.empty,
                                meta={
                                    "interrupted": content.interrupted or False,
                                },
                                eod=True,
                            )
                        )


def _speech_config(voice_name: str) -> SpeechConfigDict:
    return {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": voice_name},
        }
    }
