import json
from base64 import b64decode, b64encode, urlsafe_b64decode
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.realtime import (
    RealtimeServerEvent,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_assistant_message_param import (
    Content as AssistantContentParam,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as UserContent,
)
from openai.types.realtime.realtime_conversation_item_user_message_param import (
    Content as UserContentParam,
)

from draive import MISSING, Meta, Missing, ctx, without_missing
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputChunk,
    ModelInstructions,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    ModelOutputChunk,
    ModelSession,
    ModelSessionEvent,
    ModelSessionInput,
    ModelSessionOutput,
    ModelSessionOutputSelection,
    ModelSessionScope,
    ModelToolRequest,
    ModelToolResponse,
    ModelToolsDeclaration,
    RealtimeGenerativeModel,
)
from draive.multimodal import ArtifactContent, MultimodalContent, MultimodalContentPart, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIRealtimeConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIRealtime",)


class OpenAIRealtime(OpenAIAPI):
    def realtime_generative_model(self) -> RealtimeGenerativeModel:
        return RealtimeGenerativeModel(session_preparing=self.session_prepare)

    async def session_prepare(  # noqa: C901, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelToolsDeclaration,
        memory: ModelMemory,
        output: ModelSessionOutputSelection,
        config: OpenAIRealtimeConfig | None = None,
        **extra: Any,
    ) -> ModelSessionScope:
        config = config or ctx.state(OpenAIRealtimeConfig)
        session_config: RealtimeSessionCreateRequestParam = _prepare_session_config(
            config=config,
            instructions=instructions,
            tools=tools,
            output=output,
        )
        # managing scope manually
        scope: AbstractAsyncContextManager[str] = ctx.scope("openai_realtime")
        # prepare connection
        connection_manager: AsyncRealtimeConnectionManager = self._client.realtime.connect(
            model=config.model,
            websocket_connection_options={
                "max_size": None,  # explicitly define no size limit
            },
        )
        input_audio_format: str
        match config.input_parameters:
            case {"format": {"type": str() as audio_input}}:
                input_audio_format = audio_input

            case _:
                input_audio_format = "audio/pcm"

        output_audio_format: str
        match config.output_parameters:
            case {"format": {"type": str() as audio_input}}:
                output_audio_format = audio_input

            case _:
                output_audio_format = "audio/pcm"

        async def open_session() -> ModelSession:  # noqa: C901, PLR0915
            # enter scope
            await scope.__aenter__()
            ctx.record_info(
                attributes={
                    "model.provider": "openai",
                    "model.name": config.model,
                    "model.tools": [tool.name for tool in tools.specifications],
                    "model.tool_selection": f"{tools.selection}",
                    "model.output": f"{output}",
                },
            )
            # open connection
            connection: AsyncRealtimeConnection = await connection_manager.__aenter__()
            # setup connection
            await connection.session.update(session=session_config)

            # initialize context
            current_items: MutableMapping[str, Meta] = {}

            def message_identifier(item_id: str) -> str:
                nonlocal current_items

                if meta := current_items.get(item_id):
                    if identifier := meta.get_str("identifier"):
                        return identifier

                message_identifier: str = str(uuid4())
                current_items[item_id] = Meta.of(
                    {
                        "identifier": message_identifier,
                        "created": datetime.now(UTC).isoformat(),
                        "item_id": item_id,
                    }
                )
                return message_identifier

            memory_recall: ModelMemoryRecall = await memory.recall()
            if memory_recall.context:
                await _send_context(
                    memory_recall.context,
                    current_items=current_items,
                    connection=connection,
                )

            async def read() -> ModelSessionOutput:  # noqa: C901, PLR0911, PLR0912, PLR0915
                nonlocal current_items
                while True:
                    event: RealtimeServerEvent = await connection.recv()
                    match event.type:
                        case "response.output_audio.delta":
                            # send the audio chunk
                            return ModelOutputChunk.of(
                                ResourceContent.of(
                                    b64decode(event.delta),
                                    mime_type=output_audio_format,
                                ),
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_audio.done":
                            # send the audio end event
                            return ModelSessionEvent.of(
                                "output.audio.completed",
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_text.delta":
                            # send the text chunk
                            return ModelOutputChunk.of(
                                TextContent.of(event.delta),
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_text.done":
                            # send the text end event
                            return ModelSessionEvent.of(
                                "output.text.completed",
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_audio_transcript.delta":
                            # send the transcript text chunk (mark via meta)
                            return ModelOutputChunk.of(
                                TextContent.of(event.delta, meta={"transcript": True}),
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_audio_transcript.done":
                            # send the transcript end event
                            return ModelSessionEvent.of(
                                "output.transcript.completed",
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.done":
                            # record token usage if able - it should appear within this event
                            if usage := event.response.usage:
                                ctx.record_info(
                                    metric="model.input_tokens",
                                    value=usage.input_tokens
                                    if usage.input_tokens is not None
                                    else 0,
                                    unit="tokens",
                                    kind="counter",
                                    attributes={
                                        "model.name": config.model,
                                        "model.provider": "openai",
                                    },
                                )
                                ctx.record_info(
                                    metric="model.output_tokens",
                                    value=usage.output_tokens
                                    if usage.output_tokens is not None
                                    else 0,
                                    unit="tokens",
                                    kind="counter",
                                    attributes={
                                        "model.name": config.model,
                                        "model.provider": "openai",
                                    },
                                )

                            continue  # keep going, nothing to send here

                        case "response.output_item.done":
                            match event.item.type:
                                # received tool call
                                case "function_call":
                                    if event.item.call_id is None:
                                        continue  # can't use tool calls without call id

                                    if not event.item.name:
                                        continue  # can't use tool calls without tool name

                                    return ModelToolRequest.of(
                                        event.item.call_id,
                                        tool=event.item.name,
                                        arguments=json.loads(event.item.arguments)
                                        if event.item.arguments
                                        else None,
                                        meta={  # using predefined meta keys
                                            "created": datetime.now(UTC).isoformat(),
                                        },
                                    )

                                case "message":
                                    if event.item.id is None:
                                        continue  # can't use messages without item id

                                    if event.item.role != "assistant":
                                        continue  # skip other events

                                    # send empty eod chunk - ends the response
                                    return ModelOutputChunk.of(
                                        MultimodalContent.empty,
                                        eod=True,
                                        meta={
                                            "message_identifier": message_identifier(event.item.id),
                                            "created": datetime.now(UTC).isoformat(),
                                        },
                                    )

                                case "function_call_output":
                                    continue  # ignored for now

                                case _:
                                    continue  # ignored for now

                        case "input_audio_buffer.speech_started":
                            # send event that VAD detected input speach
                            return ModelSessionEvent.of(
                                "input.audio.started",
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "input_audio_buffer.committed":
                            # send event that input speech has ended
                            return ModelSessionEvent.of(
                                "input.audio.completed",
                                meta={
                                    "message_identifier": message_identifier(event.item_id),
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "conversation.item.created":
                            if event.item.id is None:
                                continue  # can't use items without item id

                            if event.item.type != "message":
                                continue  # skip non-message items

                            if event.item.role == "system":
                                continue  # skip system messages

                            if event.item.id in current_items:
                                continue  # we are already handling it

                            if event.item.role == "user":
                                current_items[event.item.id] = Meta.of(
                                    {  # using predefined meta keys
                                        "item_id": event.item.id,
                                        "identifier": str(uuid4()),
                                        "created": datetime.now(UTC).isoformat(),
                                    }
                                )

                            elif event.item.role == "assistant":
                                current_items[event.item.id] = Meta.of(
                                    {  # using predefined meta keys
                                        "item_id": event.item.id,
                                        "identifier": str(uuid4()),
                                        "created": datetime.now(UTC).isoformat(),
                                    }
                                )

                            # request the full item to be stored in the memory
                            await connection.conversation.item.retrieve(item_id=event.item.id)

                        case "conversation.item.input_audio_transcription.completed":
                            await connection.conversation.item.retrieve(item_id=event.item_id)

                        case "conversation.item.done":
                            if event.item.id is None:
                                continue  # can't use items without item id

                            if event.item.type != "message":
                                continue  # handle only messages

                            if event.item.role == "assistant":
                                await connection.conversation.item.retrieve(item_id=event.item.id)

                        case "conversation.item.retrieved":
                            if event.item.id is None:
                                continue  # can't use items without item id

                            if not event.item.type == "message":
                                continue  # handle only messages

                            # Only record completed items, otherwise request once more
                            if event.item.status != "completed":
                                await connection.conversation.item.retrieve(item_id=event.item.id)
                                continue  # retry getting completed event

                            assert event.item.content  # nosec: B101
                            match event.item.role:
                                case "user":
                                    try:
                                        await memory.remember(
                                            ModelInput.of(
                                                _user_content_to_multimodal(
                                                    event.item.content,
                                                    audio_format=input_audio_format,
                                                ),
                                                meta=current_items[event.item.id],
                                            ),
                                        )

                                    except Exception as exc:
                                        ctx.log_error(
                                            "Failed to remember model context",
                                            exception=exc,
                                        )
                                        raise exc

                                case "assistant":
                                    try:
                                        await memory.remember(
                                            ModelOutput.of(
                                                _assistant_content_to_multimodal(
                                                    event.item.content,
                                                    audio_format=output_audio_format,
                                                ),
                                                meta=current_items[event.item.id],
                                            ),
                                        )

                                    except Exception as exc:
                                        ctx.log_error(
                                            "Failed to remember model context",
                                            exception=exc,
                                        )
                                        raise exc

                                case _:
                                    continue  # skip other

                        case "error":
                            raise ModelException(
                                f"Realtime processing error:"
                                f" {event.error.type} - {event.error.message}",
                                provider="openai",
                                model=config.model,
                            )

                        case _:
                            continue  # skip other events

            async def write(
                input: ModelSessionInput,  # noqa: A002
            ) -> None:
                nonlocal current_items
                if isinstance(input, ModelInputChunk):
                    await _send_input_chunk(
                        input,
                        connection=connection,
                    )

                elif isinstance(input, ModelToolResponse):
                    await _send_tool_response(
                        input,
                        connection=connection,
                    )

                else:  # session event
                    match input.category:
                        case "memory.update":
                            ctx.log_debug("Received memory update event")

                            await _reset_context(
                                (await memory.recall()).context,
                                current_items=current_items,
                                connection=connection,
                            )
                            current_items = {}

                        case other:
                            ctx.log_debug(f"Received unsupported input event: {other}")

            return ModelSession(
                reading=read,
                writing=write,
            )

        async def close_session(
            exception: BaseException | None,
        ) -> None:
            nonlocal connection_manager
            await connection_manager.__aexit__(  # close connection
                type(exception) if exception is not None else None,
                exception,
                exception.__traceback__ if exception is not None else None,
            )
            await scope.__aexit__(  # exit scope
                type(exception) if exception is not None else None,
                exception,
                exception.__traceback__ if exception is not None else None,
            )

        return ModelSessionScope(
            opening=open_session,
            closing=close_session,
        )


async def _send_input_chunk(
    chunk: ModelInputChunk,
    /,
    *,
    connection: AsyncRealtimeConnection,
) -> None:
    # stream audio if we got only audio resources
    audio_parts = chunk.content.audio()
    if audio_parts and len(audio_parts) == len(chunk.content.parts):
        for part in audio_parts:
            match part:
                case ResourceContent() as media:
                    await connection.input_audio_buffer.append(audio=media.data)

                case ResourceReference():
                    # skip not supported with a log to prevent connection break
                    ctx.log_error(
                        "OpenAI realtime input (ResourceReference audio) not supported! Skipping..."
                    )

        if chunk.eod:
            await connection.input_audio_buffer.commit()

    else:
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "status": "completed" if chunk.eod else "incomplete",
                "content": _user_content_parts(chunk.content),
            },
        )


async def _send_context(
    context: ModelContext,
    /,
    *,
    current_items: MutableMapping[str, Meta],
    connection: AsyncRealtimeConnection,
) -> None:
    for element in context:
        match element:
            case ModelInput() as input_element:
                identifier: str = (input_element.meta.identifier or uuid4()).hex
                current_items[identifier] = input_element.meta
                await connection.conversation.item.create(
                    item={
                        "id": identifier,
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": _user_content_parts(
                            input_element.content,
                        ),
                    },
                )
                # include tool responses following the output
                for response in input_element.tools:
                    await connection.conversation.item.create(
                        item={
                            "id": (response.meta.identifier or uuid4()).hex,
                            "type": "function_call_output",
                            "call_id": response.identifier,
                            "output": _tool_result(response.content),
                        },
                    )

            case ModelOutput() as output_element:
                identifier: str = (output_element.meta.identifier or uuid4()).hex
                current_items[identifier] = output_element.meta
                # prior assistant content
                await connection.conversation.item.create(
                    item={
                        "id": identifier,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": _assistant_content_parts(
                            output_element.content,
                        ),
                    },
                )
                # include tool requests following the output
                for request in output_element.tools:
                    await connection.conversation.item.create(
                        item={
                            "id": (request.meta.identifier or uuid4()).hex,
                            "type": "function_call",
                            "call_id": request.identifier,
                            "name": request.tool,
                            "arguments": json.dumps(request.arguments),
                        },
                    )


async def _reset_context(
    context: ModelContext,
    /,
    current_items: MutableMapping[str, Meta],
    *,
    connection: AsyncRealtimeConnection,
) -> None:
    for item_id in current_items.keys():
        try:
            await connection.conversation.item.delete(item_id=item_id)
            del current_items[item_id]

        except Exception as exc:
            ctx.log_error(
                f"Failed to delete conversation item {item_id}",
                exception=exc,
            )

    await _send_context(
        context,
        current_items=current_items,
        connection=connection,
    )


def _user_content_parts(  # noqa: C901, PLR0912
    content: MultimodalContent,
) -> Generator[UserContentParam]:
    for part in content.parts:
        if isinstance(part, TextContent):
            if part.meta.get("transcript", False):
                yield {
                    "type": "input_text",
                    "transcript": part.text,
                }
            else:
                yield {
                    "type": "input_text",
                    "text": part.text,
                }

        elif isinstance(part, ResourceContent):
            if part.mime_type.startswith("audio"):
                # convert stored base64 (possibly urlsafe) to standard base64 string
                try:
                    raw = urlsafe_b64decode(part.data)

                except Exception:
                    raw = b64decode(part.data)

                yield {
                    "type": "input_audio",
                    "audio": b64encode(raw).decode(),
                }

            elif part.mime_type.startswith("image"):
                ctx.log_error("OpenAI realtime input (image) not supported! Skipping...")

            elif part.mime_type.startswith("video"):
                ctx.log_error("OpenAI realtime input (video) not supported! Skipping...")

            else:
                # unsupported media type
                ctx.log_error(
                    f"OpenAI realtime input (media {part.mime_type}) not supported! Skipping..."
                )

        elif isinstance(part, ResourceReference):
            # skip not supported with a log to prevent connection break
            ctx.log_error("OpenAI realtime input (ResourceReference) not supported! Skipping...")

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            yield {
                "type": "input_text",
                "text": part.artifact.to_str(),
            }


def _assistant_content_parts(
    content: MultimodalContent,
) -> Generator[AssistantContentParam]:
    for part in content.parts:
        if isinstance(part, TextContent):
            yield {
                "type": "output_text",
                "text": part.text,
            }

        elif isinstance(part, ResourceContent):
            # skip not supported with a log to prevent connection break
            ctx.log_error("OpenAI realtime output media not supported! Skipping...")

        elif isinstance(part, ResourceReference):
            # skip not supported with a log to prevent connection break
            ctx.log_error("OpenAI realtime output media not supported! Skipping...")

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            yield {
                "type": "output_text",
                "text": part.artifact.to_str(),
            }


def _tool_result(
    content: MultimodalContent,
) -> str:
    response_output: list[str] = []
    for part in content.parts:
        if isinstance(part, TextContent):
            response_output.append(part.text)

        elif isinstance(part, ResourceContent | ResourceReference):
            # skip not supported with a log to prevent connection break
            ctx.log_error("OpenAI realtime function result (media) not supported! Skipping...")

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue  # skip hidden

            response_output.append(part.artifact.to_str())

    return "".join(response_output)


async def _send_tool_response(
    response: ModelToolResponse,
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

    await connection.response.create()


def _prepare_session_config(
    *,
    instructions: ModelInstructions | None,
    config: OpenAIRealtimeConfig,
    tools: ModelToolsDeclaration,
    output: ModelSessionOutputSelection,
) -> RealtimeSessionCreateRequestParam:
    modalities: list[Literal["text", "audio"]]
    match output:
        case "auto":
            modalities = ["audio"]

        case "text":
            modalities = ["text"]

        case "audio":
            modalities = ["audio"]

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
            tool_choice = tool

    session_tools: list[Mapping[str, Any]] | Missing
    if tools:
        session_tools = [
            without_missing(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or MISSING,
                    "parameters": tool.parameters or MISSING,
                },
            )
            for tool in tools.specifications
        ]

    else:
        session_tools = MISSING

    return without_missing(
        {
            "type": "realtime",
            "model": config.model,
            "instructions": instructions if instructions is not None else MISSING,
            "audio": {
                "input": config.input_parameters,
                "output": config.output_parameters,
            },
            "output_modalities": modalities,
            "tools": session_tools,
            "tool_choice": tool_choice if session_tools is not MISSING else MISSING,
        },
        typed=RealtimeSessionCreateRequestParam,
    )


def _user_content_to_multimodal(
    content: Sequence[UserContent],
    /,
    audio_format: str,
) -> MultimodalContent:
    parts: list[MultimodalContentPart] = []
    for part in content:
        if part.text:
            parts.append(TextContent.of(part.text))

        if part.audio:
            parts.append(
                ResourceContent.of(
                    part.audio,
                    mime_type=audio_format,
                )
            )

        if part.transcript:
            parts.append(
                TextContent.of(
                    part.transcript,
                    meta={"transcript": True},
                )
            )

    return MultimodalContent.of(*parts)


def _assistant_content_to_multimodal(
    content: Sequence[AssistantContent],
    /,
    audio_format: str,
) -> MultimodalContent:
    parts: list[MultimodalContentPart] = []
    for part in content:
        if part.text:
            parts.append(TextContent.of(part.text))

        if part.audio:
            parts.append(
                ResourceContent.of(
                    part.audio,
                    mime_type=audio_format,
                )
            )

        if part.transcript:
            parts.append(
                TextContent.of(
                    part.transcript,
                    meta={"transcript": True},
                )
            )

    return MultimodalContent.of(*parts)
