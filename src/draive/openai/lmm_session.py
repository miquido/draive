import json
from asyncio import Task, TaskGroup
from base64 import b64decode, b64encode
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

from haiway import MISSING, AsyncQueue, Missing, ctx, without_missing
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.beta.realtime import (
    ConversationItemContentParam,
)
from openai.types.beta.realtime.realtime_server_event import RealtimeServerEvent
from openai.types.beta.realtime.session_update_event_param import Session, SessionTool

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
from draive.lmm.types import (
    LMMCompletion,
    LMMInput,
    LMMSessionEvent,
    LMMStreamChunk,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
)
from draive.multimodal import MediaData, MediaReference
from draive.multimodal.content import MultimodalContent
from draive.multimodal.meta import MetaContent
from draive.multimodal.text import TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIRealtimeConfig
from draive.openai.types import OpenAIException

__all__ = ("OpenAILMMSession",)


class OpenAILMMSession(OpenAIAPI):
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
        config: OpenAIRealtimeConfig | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMSessionOutput]:
        realtime_config: OpenAIRealtimeConfig = config or ctx.state(OpenAIRealtimeConfig)
        with ctx.scope("openai_realtime", realtime_config):
            ctx.record(
                attributes={
                    "lmm.provider": "openai",
                    "lmm.model": realtime_config.model,
                    "lmm.temperature": realtime_config.temperature,
                    "lmm.max_tokens": realtime_config.max_tokens,
                    "lmm.tools": [tool["name"] for tool in tools] if tools else [],
                    "lmm.tool_selection": f"{tool_selection}",
                    "lmm.output": f"{output}",
                    "lmm.context": [element.to_str() for element in initial_context or []],
                }
            )

            modalities: list[Literal["text", "audio"]]
            match output:
                case "auto":
                    modalities = ["text", "audio"]

                case "text":
                    modalities = ["text"]

                case "audio":
                    modalities = ["text", "audio"]  # openai does not allow to use only audio

                case ["text", "audio"] | ["audio", "text"]:
                    modalities = ["text", "audio"]

                case _:
                    raise ValueError(f"Unsupported output: {output}")

            tool_choice: str
            match tool_selection:
                case "auto":
                    tool_choice = "auto"

                case "required":
                    tool_choice = "required"

                case "none":
                    tool_choice = "none"

                case tool:
                    tool_choice = tool["name"]

            session_tools: list[SessionTool] | Missing
            if tools is not None:
                session_tools = [
                    without_missing(
                        {
                            "type": "function",
                            "name": tool["name"],
                            "description": tool.get("description", MISSING),
                            "parameters": tool.get("parameters", MISSING),
                        },
                        typed=SessionTool,
                    )
                    for tool in tools
                ]

            else:
                session_tools = MISSING

            output_queue = AsyncQueue[LMMSessionOutput]()
            ctx.spawn(
                self._drive_connection,
                session_connection=self._client.beta.realtime.connect(
                    model=realtime_config.model,
                ),
                session_config=without_missing(
                    {
                        "instructions": Instruction.formatted(instruction)
                        if instruction is not None
                        else MISSING,
                        "temperature": realtime_config.temperature,
                        "modalities": modalities,
                        "input_audio_format": realtime_config.input_audio_format,
                        "output_audio_format": realtime_config.output_audio_format,
                        "turn_detection": {
                            "create_response": True,
                            "eagerness": realtime_config.vad_eagerness,
                            "interrupt_response": True,
                            "type": realtime_config.vad_type,
                        }
                        if realtime_config.vad_type is not MISSING
                        else MISSING,
                        "voice": realtime_config.voice,
                        "max_response_output_tokens": realtime_config.max_tokens,
                        "tools": session_tools,
                        "tool_choice": tool_choice if session_tools is not MISSING else MISSING,
                    },
                    typed=Session,
                ),
                initial_context=initial_context,
                input_stream=input_stream,
                output_stream=output_queue,
            )

            return output_queue

    @ctx.scope("connection", task_group=TaskGroup())
    async def _drive_connection(  # noqa: C901, PLR0912, PLR0915
        self,
        session_connection: AsyncRealtimeConnectionManager,
        *,
        session_config: Session,
        initial_context: LMMContext | None,
        input_stream: AsyncIterator[LMMStreamInput],
        output_stream: AsyncQueue[LMMSessionOutput],
    ):
        input_task: Task[None] | None = None
        try:
            async with session_connection as connection:  # connect
                try:  # prepare session
                    await connection.session.update(session=session_config)

                    if initial_context:  # initialize state
                        await self._send_context(
                            initial_context,
                            connection=connection,
                        )

                    del initial_context

                except BaseException as exc:
                    return output_stream.finish(exception=exc)

                audio_output_mime: str = (
                    f"audio/{session_config.get('output_audio_format', 'pcm16')}"
                )
                # start consuming inputs
                input_task = ctx.spawn(
                    self._handle_input,
                    connection=connection,
                    input_stream=input_stream,
                    output_stream=output_stream,
                )

                # start consuming outputs
                pending_tool_calls: set[str] = set()
                partial_tool_calls: dict[str, dict[str, str]] = {}
                while not output_stream.is_finished:
                    event: RealtimeServerEvent = await connection.recv()
                    match event.type:
                        case "response.audio.delta":
                            output_stream.enqueue(
                                LMMStreamChunk.of(
                                    MediaData.of(
                                        b64decode(event.delta),
                                        media=audio_output_mime,
                                    )
                                )
                            )

                        case "response.audio.done":
                            output_stream.enqueue(LMMSessionEvent.of("completed"))

                        case "response.function_call_arguments.delta":
                            partial_tool_calls[event.item_id]["arguments"] += event.delta

                        case "response.function_call_arguments.done":
                            tool_call: Mapping[str, str] = partial_tool_calls[event.item_id]
                            del partial_tool_calls[event.item_id]
                            pending_tool_calls.add(tool_call["call_id"])
                            output_stream.enqueue(
                                LMMToolRequest.of(
                                    tool_call["call_id"],
                                    tool=tool_call["tool"],
                                    arguments=json.loads(tool_call["arguments"])
                                    if tool_call["arguments"]
                                    else None,
                                )
                            )

                        case "response.text.delta":
                            output_stream.enqueue(LMMStreamChunk.of(event.delta))

                        case "response.text.done":
                            output_stream.enqueue(LMMSessionEvent.of("completed"))

                        case "response.output_item.added":
                            if event.item.type != "function_call":
                                continue  # skip other items

                            if event.item.id is None:
                                continue  # can't use events without item ids

                            if event.item.call_id is None:
                                continue  # skip tool calls without call id

                            if event.item.name is None:
                                continue  # skip tool calls without tool name

                            match event.item.status:
                                case "completed":
                                    pending_tool_calls.add(event.item.call_id)
                                    output_stream.enqueue(
                                        LMMToolRequest.of(
                                            event.item.call_id,
                                            tool=event.item.name,
                                            arguments=json.loads(event.item.arguments)
                                            if event.item.arguments
                                            else None,
                                        )
                                    )

                                case _:
                                    partial_tool_calls[event.item.id] = {
                                        "call_id": event.item.call_id,
                                        "tool": event.item.name,
                                        "arguments": event.item.arguments or "",
                                    }

                        case "conversation.item.created":
                            if event.item.type != "function_call_output":
                                continue  # skip other items

                            assert event.item.call_id is not None  # nosec: B101

                            if event.item.call_id not in pending_tool_calls:
                                continue  # skip tools which are not expected to respond

                            pending_tool_calls.remove(event.item.call_id)

                            if not pending_tool_calls:
                                # request a response after providing all tool results
                                await connection.response.create()

                        case "input_audio_buffer.speech_started":
                            # treat audio input start as an interrupt
                            output_stream.enqueue(LMMSessionEvent.of("interrupted"))
                            # skip pending tool results in that case?
                            pending_tool_calls.clear()

                        case "error":
                            raise OpenAIException(
                                f"Realtime processing error:"
                                f" {event.error.type} - {event.error.message}",
                                event,
                            )

                        case _:
                            pass  # skip other events

        except BaseException as exc:
            output_stream.finish(exc)

        else:
            output_stream.finish()

        finally:
            if input_task is not None:
                input_task.cancel()

    async def _send_input_chunk(
        self,
        chunk: LMMStreamChunk,
        /,
        *,
        connection: AsyncRealtimeConnection,
    ) -> None:
        # stream audio if we got only audio
        if chunk.content.is_media("audio"):
            for part in chunk.content.media("audio"):
                match part:
                    case MediaData() as media:
                        await connection.input_audio_buffer.append(
                            audio=b64encode(media.data).decode()
                        )

                    case _:
                        # skip not supported with a log to prevent connection break
                        ctx.log_error(
                            "OpenAI realtime input (MediaReference) not supported! Skipping..."
                        )

            if chunk.eod:
                await connection.input_audio_buffer.commit()

        else:
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "status": "completed" if chunk.eod else "incomplete",
                    "content": self._item_content_parts(chunk.content),
                },
            )

    async def _send_context(
        self,
        context: LMMContext,
        /,
        *,
        connection: AsyncRealtimeConnection,
    ) -> None:
        for element in context:
            match element:
                case LMMInput() as input_element:
                    await connection.conversation.item.create(
                        item={
                            "type": "message",
                            "status": "completed",
                            "role": "user",
                            "content": self._item_content_parts(input_element.content),
                        },
                    )

                case LMMCompletion() as completion_element:
                    await connection.conversation.item.create(
                        item={
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": self._item_content_parts(completion_element.content),
                        },
                    )

                case LMMToolRequests() as tool_requests:
                    for request in tool_requests.requests:
                        await connection.conversation.item.create(
                            item={
                                "type": "function_call",
                                "call_id": request.identifier,
                                "arguments": json.dumps(request.arguments),
                            },
                        )

                case LMMToolResponses() as tool_responses:
                    for response in tool_responses.responses:
                        await connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": response.identifier,
                                "output": self._tool_result(response.content),
                            },
                        )

    def _item_content_parts(
        self,
        content: MultimodalContent,
    ) -> list[ConversationItemContentParam]:
        content_parts: list[ConversationItemContentParam] = []
        for part in content.parts:
            match part:
                case TextContent() as text:
                    content_parts.append(
                        {
                            "type": "input_text",
                            "text": text.text,
                        }
                    )

                case MediaData() as media:
                    match media.media:
                        case "audio":
                            content_parts.append(
                                {
                                    "type": "input_audio",
                                    "audio": b64encode(media.data).decode(),
                                }
                            )

                        case "image":
                            # skip not supported with a log to prevent connection break
                            ctx.log_error(
                                "OpenAI realtime input (image) not supported! Skipping..."
                            )

                        case "video":
                            # skip not supported with a log to prevent connection break
                            ctx.log_error(
                                "OpenAI realtime input (video) not supported! Skipping..."
                            )

                case MediaReference():
                    # skip not supported with a log to prevent connection break
                    ctx.log_error(
                        "OpenAI realtime input (MediaReference) not supported! Skipping..."
                    )

                case MetaContent():
                    # skip not supported with a log to prevent connection break
                    ctx.log_error("OpenAI realtime input (MetaContent) not supported! Skipping...")

                case other:  # treat other as json text
                    content_parts.append(
                        {
                            "type": "input_text",
                            "text": other.to_json(),
                        }
                    )

        return content_parts

    def _tool_result(
        self,
        content: MultimodalContent,
    ) -> str:
        response_output: str = ""
        for part in content.parts:
            match part:
                case TextContent() as text:
                    response_output += text.text

                case MediaData() | MediaReference():
                    # skip not supported with a log to prevent connection break
                    ctx.log_error(
                        "OpenAI realtime function result (media) not supported! Skipping..."
                    )

                case MetaContent():
                    # skip not supported with a log to prevent connection break
                    ctx.log_error(
                        "OpenAI realtime function result (MetaContent) not supported! Skipping..."
                    )

                case other:  # treat other as json text
                    response_output += other.to_json()

        return response_output

    async def _send_tool_response(
        self,
        response: LMMToolResponse,
        /,
        *,
        connection: AsyncRealtimeConnection,
    ) -> None:
        await connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": response.identifier,
                "output": self._tool_result(response.content),
            },
        )

    async def _handle_input(
        self,
        connection: AsyncRealtimeConnection,
        *,
        input_stream: AsyncIterator[LMMStreamInput],
        output_stream: AsyncQueue[LMMSessionOutput],
    ) -> None:
        try:
            async for element in input_stream:
                match element:
                    case LMMStreamChunk() as chunk:
                        await self._send_input_chunk(
                            chunk,
                            connection=connection,
                        )

                    case LMMToolResponse() as tool_response:
                        await self._send_tool_response(
                            tool_response,
                            connection=connection,
                        )

        except BaseException as exc:
            output_stream.finish(exception=exc)

        else:
            output_stream.finish()
