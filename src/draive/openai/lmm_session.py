import json
from base64 import b64decode, b64encode
from collections.abc import Sequence
from datetime import datetime
from types import TracebackType
from typing import Any, Literal
from uuid import uuid4

from haiway import MISSING, Missing, ObservabilityLevel, ScopeContext, ctx, without_missing
from openai.resources.beta.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.beta.realtime import (
    ConversationItemContent,
    ConversationItemContentParam,
    RealtimeServerEvent,
)
from openai.types.beta.realtime.session_update_event_param import Session, SessionTool

from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMInput,
    LMMInstruction,
    LMMMemory,
    LMMSession,
    LMMSessionEvent,
    LMMSessionInput,
    LMMSessionOutput,
    LMMSessionOutputSelection,
    LMMSessionScope,
    LMMStreamChunk,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
    LMMTools,
    RealtimeLMM,
)
from draive.multimodal import (
    MediaData,
    MediaReference,
    MetaContent,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIRealtimeConfig
from draive.openai.types import OpenAIException
from draive.utils import MEMORY_NONE

__all__ = ("OpenAIRealtimeLMM",)


class OpenAIRealtimeLMM(OpenAIAPI):
    def lmm_session(self) -> RealtimeLMM:
        return RealtimeLMM(session_preparing=self.lmm_session_prepare)

    async def lmm_session_prepare(  # noqa: C901, PLR0915
        self,
        *,
        instruction: LMMInstruction | None = None,
        memory: LMMMemory | None = None,
        tools: LMMTools | None = None,
        output: LMMSessionOutputSelection = "auto",
        config: OpenAIRealtimeConfig | None = None,
        **extra: Any,
    ) -> LMMSessionScope:
        # prepare config
        tools = tools or LMMTools.none
        memory = memory if memory is not None else MEMORY_NONE
        config = config or ctx.state(OpenAIRealtimeConfig)
        session_config: Session = _prepare_session_config(
            config=config,
            instruction=instruction,
            tools=tools,
            output=output,
        )
        # managing scope manually
        scope: ScopeContext = ctx.scope("openai_realtime", config)
        # prepare connection
        connection_manager: AsyncRealtimeConnectionManager = self._client.beta.realtime.connect(
            model=config.model,
        )
        # prepare output audio format
        output_audio_type: str = f"audio/{config.output_audio_format}"

        async def open_session() -> LMMSession:  # noqa: C901, PLR0915
            # enter scope
            await scope.__aenter__()
            ctx.record(
                attributes={
                    "lmm.provider": "openai",
                    "lmm.model": config.model,
                    "lmm.transcribe_model": config.transcribe_model,
                    "lmm.input_audio_noise_reduction": config.input_audio_noise_reduction,
                    "lmm.vad_type": config.vad_type,
                    "lmm.vad_eagerness": config.vad_eagerness,
                    "lmm.voice": config.voice,
                    "lmm.tools": [tool["name"] for tool in tools.specifications],
                    "lmm.tool_selection": f"{tools.selection}",
                    "lmm.output": f"{output}",
                }
            )
            # open connection
            connection: AsyncRealtimeConnection = await connection_manager.__aenter__()
            # setup connection
            await connection.session.update(session=session_config)

            # initialize context
            context: LMMContext = await memory.recall()
            if context:
                await _send_context(
                    context,
                    connection=connection,
                )

            # track tools usage
            pending_tool_calls: dict[str, tuple[LMMToolRequest, LMMToolResponse | None]] = {}

            input_audio_format: str = f"audio/{config.input_audio_format}"
            output_audio_format: str = f"audio/{config.output_audio_format}"

            async def read() -> LMMSessionOutput:  # noqa: C901, PLR0911, PLR0912, PLR0915
                nonlocal pending_tool_calls
                while True:
                    event: RealtimeServerEvent = await connection.recv()
                    match event.type:
                        # received response audio chunk
                        case "response.audio.delta":
                            return LMMStreamChunk.of(
                                MediaData.of(
                                    b64decode(event.delta),
                                    media=output_audio_type,
                                ),
                                meta={
                                    "origin_identifier": event.response_id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        # response audio completed
                        case "response.audio.done":
                            return LMMSessionEvent.of(
                                "output.audio.completed",
                                meta={
                                    "origin_identifier": event.response_id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        # received response text chunk
                        case "response.text.delta":
                            return LMMStreamChunk.of(
                                TextContent.of(event.delta),
                                meta={
                                    "origin_identifier": event.response_id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        # response text completed
                        case "response.text.done":
                            return LMMSessionEvent.of(
                                "output.text.completed",
                                meta={
                                    "origin_identifier": event.response_id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        # response completed
                        case "response.done":
                            # record token usage if able
                            if usage := event.response.usage:
                                ctx.record(
                                    ObservabilityLevel.INFO,
                                    metric="lmm.input_tokens",
                                    value=usage.input_tokens
                                    if usage.input_tokens is not None
                                    else 0,
                                    unit="tokens",
                                    attributes={"lmm.model": config.model},
                                )
                                ctx.record(
                                    ObservabilityLevel.INFO,
                                    metric="lmm.output_tokens",
                                    value=usage.output_tokens
                                    if usage.output_tokens is not None
                                    else 0,
                                    unit="tokens",
                                    attributes={"lmm.model": config.model},
                                )

                            if event.response.status == "incomplete":
                                continue  # TODO: handle incomplete if needed

                            # record response in memory
                            if event.response.output:
                                await memory.remember(
                                    LMMCompletion.of(
                                        MultimodalContent.empty,
                                        meta={
                                            "origin_identifier": event.response.id,
                                            "created": datetime.now().isoformat(),
                                        },
                                    )
                                )

                            # send empty eod chunk
                            return LMMStreamChunk.of(
                                MultimodalContent.empty,
                                eod=True,
                                meta={
                                    "origin_identifier": event.response.id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        case "response.output_item.done":
                            match event.item.type:
                                # tool call received
                                case "function_call":
                                    if event.item.call_id is None:
                                        continue  # can't use tool calls without call id

                                    if event.item.name is None:
                                        continue  # can't use tool calls without tool name

                                    request: LMMToolRequest = LMMToolRequest.of(
                                        event.item.call_id,
                                        tool=event.item.name,
                                        arguments=json.loads(event.item.arguments)
                                        if event.item.arguments
                                        else None,
                                        meta={
                                            "origin_identifier": event.item.id,
                                            "created": datetime.now().isoformat(),
                                        },
                                    )
                                    pending_tool_calls[request.identifier] = (request, None)
                                    return request

                                case "message":
                                    if event.item.role != "assistant":
                                        continue  # skip other events

                                    if event.item.content:
                                        await memory.remember(
                                            LMMCompletion.of(
                                                _content_to_multimodal(
                                                    event.item.content,
                                                    audio_format=output_audio_format,
                                                ),
                                                meta={
                                                    "origin_identifier": event.item.id,
                                                    "created": datetime.now().isoformat(),
                                                },
                                            ),
                                        )

                                    continue  # keep running

                                case "function_call_output":
                                    continue  # ignored - we are producing call results manually

                        # input has beed added
                        case "conversation.item.created":
                            match event.item.type:
                                # added message
                                case "message":
                                    if event.item.role != "user":
                                        continue  # skip other events

                                    if event.item.content:
                                        await memory.remember(
                                            LMMInput.of(
                                                _content_to_multimodal(
                                                    event.item.content,
                                                    audio_format=input_audio_format,
                                                ),
                                                meta={
                                                    "origin_identifier": event.item.id,
                                                    "created": datetime.now().isoformat(),
                                                },
                                            ),
                                        )

                                    return LMMSessionEvent.of(
                                        "input.completed",
                                        meta={
                                            "origin_identifier": event.item.id,
                                            "created": datetime.now().isoformat(),
                                        },
                                    )

                                # added function result
                                case "function_call_output":
                                    # check if we got all responses
                                    if any(call[1] is None for call in pending_tool_calls.values()):
                                        continue  # wait for all responses

                                    # prepare memory
                                    requests: list[LMMToolRequest] = []
                                    responses: list[LMMToolResponse] = []
                                    for request, response in pending_tool_calls.values():
                                        assert response is not None  # nosec: B101
                                        requests.append(request)
                                        responses.append(response)

                                    # remember requests and responses
                                    await memory.remember(
                                        LMMToolRequests.of(
                                            requests,
                                            # TODO: we are missing extra content
                                            # added with tool calls
                                            content=MultimodalContent.empty,
                                        ),
                                        LMMToolResponses.of(responses),
                                    )
                                    # cleanup pending tool calls
                                    pending_tool_calls = {}
                                    # request a response after providing all tool results
                                    await connection.response.create()

                                case "function_call":
                                    continue  # ignored - we are not producing calls manually

                        case "input_audio_buffer.speech_started":
                            return LMMSessionEvent.of(
                                "input.audio.started",
                                meta={
                                    "origin_identifier": event.item_id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        case "input_audio_buffer.committed":
                            return LMMSessionEvent.of(
                                "input.audio.completed",
                                meta={
                                    "origin_identifier": event.item_id,
                                    "created": datetime.now().isoformat(),
                                },
                            )

                        case "conversation.item.input_audio_transcription.completed":
                            continue  # TODO: we should handle it in memory, not send to output

                        case "error":
                            raise OpenAIException(
                                f"Realtime processing error:"
                                f" {event.error.type} - {event.error.message}",
                                event,
                            )

                        case _:
                            continue  # skip other events

            async def write(
                input: LMMSessionInput,  # noqa: A002
            ) -> None:
                match input:
                    case LMMStreamChunk() as chunk:
                        await _send_input_chunk(
                            chunk,
                            connection=connection,
                        )

                    case LMMToolResponse() as tool_response:
                        assert pending_tool_calls.get(tool_response.identifier) is not None  # nosec: B101
                        assert pending_tool_calls[tool_response.identifier][1] is None  # nosec: B101
                        # TODO: consider storing single requests/responses instead of manually
                        # managining it and storing in batches, also ensure removing unhandled
                        # record response
                        pending_tool_calls[tool_response.identifier] = (
                            pending_tool_calls[tool_response.identifier][0],
                            tool_response,
                        )
                        await _send_tool_response(
                            tool_response,
                            connection=connection,
                        )

            return LMMSession(
                reading=read,
                writing=write,
            )

        async def close_session(
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            nonlocal connection_manager
            await connection_manager.__aexit__(  # close connection
                exc_type,
                exc_val,
                exc_tb,
            )
            await scope.__aexit__(  # exit scope
                exc_type,
                exc_val,
                exc_tb,
            )

        return LMMSessionScope(
            opening=open_session,
            closing=close_session,
        )


async def _send_input_chunk(
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
                    await connection.input_audio_buffer.append(audio=b64encode(media.data).decode())

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
                "content": _item_content_parts(chunk.content),
            },
        )


async def _send_context(
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
                        "id": (input_element.meta.identifier or uuid4()).hex,
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": _item_content_parts(input_element.content),
                    },
                )

            case LMMCompletion() as completion_element:
                await connection.conversation.item.create(
                    item={
                        "id": (completion_element.meta.identifier or uuid4()).hex,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": _item_content_parts(completion_element.content),
                    },
                )

            case LMMToolRequests() as tool_requests:
                for request in tool_requests.requests:
                    await connection.conversation.item.create(
                        item={
                            "id": (request.meta.identifier or uuid4()).hex,
                            "type": "function_call",
                            "call_id": request.identifier,
                            "arguments": json.dumps(request.arguments),
                        },
                    )

            case LMMToolResponses() as tool_responses:
                for response in tool_responses.responses:
                    await connection.conversation.item.create(
                        item={
                            "id": (response.meta.identifier or uuid4()).hex,
                            "type": "function_call_output",
                            "call_id": response.identifier,
                            "output": _tool_result(response.content),
                        },
                    )


def _item_content_parts(
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
                        ctx.log_error("OpenAI realtime input (image) not supported! Skipping...")

                    case "video":
                        # skip not supported with a log to prevent connection break
                        ctx.log_error("OpenAI realtime input (video) not supported! Skipping...")

            case MediaReference():
                # skip not supported with a log to prevent connection break
                ctx.log_error("OpenAI realtime input (MediaReference) not supported! Skipping...")

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
    content: MultimodalContent,
) -> str:
    response_output: str = ""
    for part in content.parts:
        match part:
            case TextContent() as text:
                response_output += text.text

            case MediaData() | MediaReference():
                # skip not supported with a log to prevent connection break
                ctx.log_error("OpenAI realtime function result (media) not supported! Skipping...")

            case MetaContent():
                # skip not supported with a log to prevent connection break
                ctx.log_error(
                    "OpenAI realtime function result (MetaContent) not supported! Skipping..."
                )

            case other:  # treat other as json text
                response_output += other.to_json()

    return response_output


async def _send_tool_response(
    response: LMMToolResponse,
    /,
    *,
    connection: AsyncRealtimeConnection,
) -> None:
    await connection.conversation.item.create(
        item={
            "type": "function_call_output",
            "call_id": response.identifier,
            "output": _tool_result(response.content),
        },
    )


def _prepare_session_config(
    *,
    instruction: LMMInstruction | None,
    config: OpenAIRealtimeConfig,
    tools: LMMTools,
    output: LMMSessionOutputSelection,
) -> Session:
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
    match tools.selection:
        case "auto":
            tool_choice = "auto"

        case "required":
            tool_choice = "required"

        case "none":
            tool_choice = "none"

        case tool:
            tool_choice = tool["name"]

    session_tools: list[SessionTool] | Missing
    if tools:
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
            for tool in tools.specifications
        ]

    else:
        session_tools = MISSING

    return without_missing(
        {
            "instructions": instruction if instruction is not None else MISSING,
            "modalities": modalities,
            "input_audio_format": config.input_audio_format,
            "output_audio_format": config.output_audio_format,
            "turn_detection": {
                "create_response": True,
                "eagerness": config.vad_eagerness,
                "interrupt_response": True,
                "type": config.vad_type,
            }
            if config.vad_type is not MISSING
            else MISSING,
            "voice": config.voice,
            "tools": session_tools,
            "tool_choice": tool_choice if session_tools is not MISSING else MISSING,
            "input_audio_noise_reduction": {
                "type": config.input_audio_noise_reduction,
            }
            if config.input_audio_noise_reduction is not MISSING
            else MISSING,
            "input_audio_transcription": {
                "model": config.transcribe_model,
            }
            if config.transcribe_model is not MISSING
            else MISSING,
        },
        typed=Session,
    )


def _content_to_multimodal(
    content: Sequence[ConversationItemContent],
    /,
    audio_format: str,
) -> MultimodalContent:
    parts: list[MultimodalContentElement] = []
    for part in content:
        if part.text:
            parts.append(TextContent.of(part.text))

        if part.audio:
            parts.append(
                MediaData.of(
                    part.audio,
                    media=audio_format,
                )
            )

    return MultimodalContent.of(*parts)
