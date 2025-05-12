# LMMSession
from asyncio import Task, TaskGroup
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from itertools import chain
from typing import Any

from google.genai.live import AsyncSession
from google.genai.types import (
    Content,
    ContentDict,
    FunctionCallingConfigMode,
    FunctionDeclarationDict,
    Modality,
    SchemaDict,
    SpeechConfigDict,
)
from haiway import AsyncQueue, as_list, ctx

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
    LMMSessionOutput,
    LMMSessionOutputSelection,
    LMMStreamInput,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.lmm.types import LMMSessionEvent, LMMStreamChunk, LMMToolRequest, LMMToolResponse
from draive.multimodal import MediaData, MediaReference
from draive.multimodal.content import MultimodalContent
from draive.multimodal.text import TextContent
from draive.parameters import DataModel

__all__ = ("GeminiLMMSession",)


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
    ) -> AsyncIterator[LMMSessionOutput]:
        live_config: GeminiLiveConfig = config or ctx.state(GeminiLiveConfig)
        with ctx.scope("gemini_live", live_config):
            ctx.record(
                attributes={
                    "lmm.provider": "gemini",
                    "lmm.model": live_config.model,
                    "lmm.temperature": live_config.temperature,
                    "lmm.max_tokens": live_config.max_tokens,
                    "lmm.seed": live_config.seed,
                    "lmm.tools": [tool["name"] for tool in tools] if tools else [],
                    "lmm.tool_selection": f"{tool_selection}",
                    "lmm.output": f"{output}",
                    "lmm.context": [element.to_str() for element in initial_context or []],
                }
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

            output_queue = AsyncQueue[LMMSessionOutput]()

            ctx.spawn(
                self._drive_connection,
                session_connection=self._client.aio.live.connect(
                    model=live_config.model,
                    config={
                        "system_instruction": {
                            "parts": [{"text": Instruction.formatted(instruction)}]
                        },
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
                ),
                input_content=initial_content,
                input_stream=input_stream,
                output_stream=output_queue,
            )

            return output_queue

    @ctx.scope("connection", task_group=TaskGroup())
    async def _drive_connection(  # noqa: C901, PLR0912
        self,
        session_connection: AbstractAsyncContextManager[AsyncSession],
        input_content: list[Content | ContentDict],
        input_stream: AsyncIterator[LMMStreamInput],
        output_stream: AsyncQueue[LMMSessionOutput],
    ):
        try:
            async with session_connection as session:
                input_task: Task[None] = ctx.spawn(
                    self._handle_input,
                    session=session,
                    input_stream=input_stream,
                    output_stream=output_stream,
                )
                try:
                    if input_content:
                        await session.send_client_content(
                            turns=input_content,
                            turn_complete=True,
                        )

                    del input_content

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
                                            eod=eod,
                                        )
                                    )

                                if content.interrupted or False:
                                    output_stream.enqueue(LMMSessionEvent.of("interrupted"))

                                elif eod:
                                    output_stream.enqueue(LMMSessionEvent.of("completed"))
                finally:
                    input_task.cancel()

        except BaseException as exc:
            output_stream.finish(exc)

        else:
            output_stream.finish()

    async def _handle_output(  # noqa: C901
        self,
        session: AsyncSession,
        output_stream: AsyncQueue[LMMSessionOutput],
    ) -> None:
        try:
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
                                    eod=eod,
                                )
                            )

                        if content.interrupted or False:
                            output_stream.enqueue(LMMSessionEvent.of("interrupted"))

                        elif eod:
                            output_stream.enqueue(LMMSessionEvent.of("completed"))

        except BaseException as exc:
            output_stream.finish(exception=exc)

        else:
            output_stream.finish()

    async def _handle_input(  # noqa: C901
        self,
        session: AsyncSession,
        input_stream: AsyncIterator[LMMStreamInput],
        output_stream: AsyncQueue[LMMSessionOutput],
    ) -> None:
        try:
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

                                case MediaData() as media_data:
                                    await session.send_realtime_input(
                                        media={
                                            "data": media_data.data,
                                            "mime_type": media_data.media,
                                        },
                                    )

                                case MediaReference():
                                    raise ValueError("Media reference is not supported")

                                case DataModel() as data:
                                    await session.send_realtime_input(
                                        media={
                                            "data": data.to_json().encode(),
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
                                if tool_response.handling == "error"
                                else {
                                    "output": [
                                        content_element_as_part(element)
                                        for element in tool_response.content.parts
                                    ],
                                },
                            }
                        )

        except BaseException as exc:
            output_stream.finish(exception=exc)

        else:
            output_stream.finish()


def _speech_config(voice_name: str) -> SpeechConfigDict:
    return {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": voice_name},
        }
    }
