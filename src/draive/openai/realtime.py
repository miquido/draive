import json
from base64 import b64decode, b64encode, urlsafe_b64decode
from collections.abc import Generator, Mapping, MutableMapping
from contextlib import AbstractAsyncContextManager
from copy import copy
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from haiway import MISSING, Meta, Missing, State, ctx, without_missing
from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.realtime import (
    RealtimeServerEvent,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_conversation_item_assistant_message_param import (
    Content as AssistantContentParam,
)
from openai.types.realtime.realtime_conversation_item_user_message_param import (
    Content as UserContentParam,
)

from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelSession,
    ModelSessionEvent,
    ModelSessionInputChunk,
    ModelSessionOutputChunk,
    ModelSessionOutputSelection,
    ModelSessionScope,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.models.metrics import record_model_invocation, record_usage_metrics
from draive.multimodal import ArtifactContent, MultimodalContent, MultimodalContentPart, TextContent
from draive.openai.api import OpenAIAPI
from draive.openai.config import OpenAIRealtimeConfig
from draive.resources import ResourceContent, ResourceReference

__all__ = ("OpenAIRealtime",)


class OpenAIRealtime(OpenAIAPI):
    async def session_prepare(  # noqa: C901, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
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
        scope: AbstractAsyncContextManager[str] = ctx.scope("model.session")
        # prepare connection
        connection_manager: AsyncRealtimeConnectionManager = self._client.realtime.connect(
            model=config.model,
            websocket_connection_options={
                "max_size": None,  # explicitly define no size limit
            },
        )

        output_audio_format: str
        match config.output_parameters:
            case {"format": {"type": str() as audio_input}}:
                output_audio_format = audio_input

            case _:
                output_audio_format = "audio/pcm"

        async def open_session() -> ModelSession:  # noqa: C901, PLR0915
            # enter scope
            await scope.__aenter__()
            record_model_invocation(
                provider="openai",
                model=config.model,
                tools=tools,
                output=output,
            )
            # open connection
            connection: AsyncRealtimeConnection = await connection_manager.__aenter__()
            # setup connection
            await connection.session.update(session=session_config)

            current_items: MutableMapping[str, Meta] = {}

            if context:
                await _send_context(
                    context,
                    current_items=current_items,
                    connection=connection,
                )

            async def read() -> ModelSessionOutputChunk:  # noqa: C901, PLR0911, PLR0912, PLR0915
                nonlocal current_items
                while True:
                    event: RealtimeServerEvent = await connection.recv()
                    match event.type:
                        case "response.output_audio.delta":
                            # send the audio chunk
                            return ResourceContent.of(
                                b64decode(event.delta),
                                mime_type=output_audio_format,
                                meta={
                                    "identifier": event.item_id,
                                    "item_id": event.item_id,
                                    "response_id": event.response_id,
                                    "output_index": event.output_index,
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_text.delta":
                            # send the text chunk
                            return TextContent.of(
                                event.delta,
                                meta={
                                    "identifier": event.item_id,
                                    "item_id": event.item_id,
                                    "response_id": event.response_id,
                                    "output_index": event.output_index,
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "response.output_item.done":
                            assert event.item.id is not None  # nosec: B101

                            match event.item.type:
                                # received tool call
                                case "function_call":
                                    assert event.item.status == "completed"  # nosec: B101

                                    return ModelToolRequest.of(
                                        event.item.call_id or str(uuid4()),
                                        tool=event.item.name,
                                        arguments=json.loads(event.item.arguments)
                                        if event.item.arguments
                                        else None,
                                        meta={  # using predefined meta keys
                                            "identifier": event.item.id,
                                            "item_id": event.item.id,
                                            "call_id": event.item.call_id,
                                            "response_id": event.response_id,
                                            "output_index": event.output_index,
                                            "created": datetime.now(UTC).isoformat(),
                                        },
                                    )

                                case "message":
                                    if event.item.role != "assistant":
                                        continue  # skip other events

                                    # send eod event - ends the response
                                    return ModelSessionEvent.turn_finished(
                                        meta={
                                            "identifier": event.item.id,
                                            "item_id": event.item.id,
                                            "response_id": event.response_id,
                                            "output_index": event.output_index,
                                        },
                                    )

                                case _:
                                    continue  # ignored for now

                        case "input_audio_buffer.speech_started":
                            # send event that VAD detected input speach
                            return ModelSessionEvent.turn_started(
                                meta={
                                    "identifier": event.item_id,
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "input_audio_buffer.committed":
                            # send event that input speech has ended
                            return ModelSessionEvent.turn_commited(
                                meta={
                                    "identifier": event.item_id,
                                    "created": datetime.now(UTC).isoformat(),
                                },
                            )

                        case "conversation.item.created":
                            assert event.item.id is not None  # nosec: B101

                            if event.item.type != "message":
                                continue  # skip non-message items

                            if event.item.id in current_items:
                                continue  # we are already handling it

                            if event.item.role == "user":
                                current_items[event.item.id] = Meta.of(
                                    {  # using predefined meta keys
                                        "identifier": event.item.id,
                                        "created": datetime.now(UTC).isoformat(),
                                    }
                                )
                                # request the full item to be stored in the memory
                                await connection.conversation.item.retrieve(item_id=event.item.id)

                            elif event.item.role == "assistant":
                                current_items[event.item.id] = Meta.of(
                                    {  # using predefined meta keys
                                        "identifier": event.item.id,
                                        "created": datetime.now(UTC).isoformat(),
                                    }
                                )
                                # request the full item to be stored in the memory
                                await connection.conversation.item.retrieve(item_id=event.item.id)

                            else:
                                continue  # skip other

                        case "response.done":
                            # record token usage if able - it should appear within this event
                            if usage := event.response.usage:
                                record_usage_metrics(
                                    provider="openai",
                                    model=config.model,
                                    input_tokens=usage.input_tokens,
                                    cached_input_tokens=(
                                        usage.input_token_details.cached_tokens
                                        if usage.input_token_details is not None
                                        else None
                                    ),
                                    output_tokens=usage.output_tokens,
                                )

                            continue  # keep going, nothing to send here

                        case "conversation.item.input_audio_transcription.completed":
                            # request the full item to be stored in the memory
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

                            if event.item.type != "message":
                                continue  # handle only messages

                            # Only record completed items, otherwise request once more
                            if event.item.status != "completed":
                                await connection.conversation.item.retrieve(item_id=event.item.id)
                                continue  # retry getting completed event

                            assert event.item.content  # nosec: B101

                            if event.item.role == "user":
                                item_meta: Meta = current_items.get(
                                    event.item.id,
                                    Meta.of(
                                        {
                                            "identifier": event.item.id,
                                            "created": datetime.now(UTC).isoformat(),
                                        }
                                    ),
                                )

                                return ModelSessionEvent.turn_completed(
                                    ModelInput.of(
                                        MultimodalContent.of(
                                            *_content_to_multimodal(
                                                event.item.content,
                                                audio_format="audio/pcm",
                                            )
                                        ),
                                        meta=item_meta,
                                    ),
                                    meta=item_meta,
                                )

                            elif event.item.role == "assistant":
                                item_meta: Meta = current_items.get(
                                    event.item.id,
                                    Meta.of(
                                        {
                                            "identifier": event.item.id,
                                            "created": datetime.now(UTC).isoformat(),
                                        }
                                    ),
                                )

                                return ModelSessionEvent.turn_completed(
                                    ModelOutput.of(
                                        MultimodalContent.of(
                                            *_content_to_multimodal(
                                                event.item.content,
                                                audio_format=output_audio_format,
                                            )
                                        ),
                                        meta=item_meta,
                                    ),
                                    meta=item_meta,
                                )

                            else:
                                continue  # skip other items

                        case "error":
                            raise ModelException(
                                f"Realtime processing error:"
                                f" {event.error.type} - {event.error.message}",
                                provider="openai",
                                model=config.model,
                            )

                        case _:
                            continue  # skip other events

            async def send_input_part(
                part: MultimodalContentPart,
                /,
            ) -> None:
                if isinstance(part, ResourceContent):
                    await connection.input_audio_buffer.append(audio=part.data)

                else:
                    ctx.log_error("OpenAI realtime input not supported! Skipping...")

            async def write(
                input: ModelSessionInputChunk,  # noqa: A002
            ) -> None:
                if isinstance(input, MultimodalContentPart):
                    await send_input_part(input)

                elif isinstance(input, ModelToolResponse):
                    await _send_tool_response(
                        input,
                        connection=connection,
                    )

                else:
                    assert isinstance(input, ModelSessionEvent)  # nosec: B101
                    if input.event == "turn_commited":
                        await connection.input_audio_buffer.commit()

                    elif input.event == "context_updated":
                        ctx.log_debug("Context memory update event")
                        await _reset_context(
                            _event_context(input),
                            current_items=current_items,
                            connection=connection,
                        )

                    else:
                        ctx.log_debug(f"Received unsupported input event: {input.event}")

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
                identifier: str = str(input_element.meta.identifier or uuid4())
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
                for response in input_element.tool_responses:
                    item_id: str = str(response.meta.identifier or uuid4())
                    current_items[item_id] = Meta.of(
                        {
                            "item_id": item_id,
                            "identifier": item_id,
                            "created": datetime.now(UTC).isoformat(),
                        }
                    )
                    await connection.conversation.item.create(
                        item={
                            "id": item_id,
                            "type": "function_call_output",
                            "call_id": response.identifier,
                            "output": _tool_result(response.result),
                        },
                    )

            case ModelOutput() as output_element:
                identifier: str = str(output_element.meta.identifier or uuid4())
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
                for request in output_element.tool_requests:
                    item_id: str = str(request.meta.identifier or uuid4())
                    current_items[item_id] = Meta.of(
                        {
                            "item_id": item_id,
                            "identifier": item_id,
                            "created": datetime.now(UTC).isoformat(),
                        }
                    )
                    await connection.conversation.item.create(
                        item={
                            "id": item_id,
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
    for item_id in copy(current_items).keys():
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
            if part.meta.get("transcript"):
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
                raw_data: bytes
                try:
                    raw_data = urlsafe_b64decode(part.data)

                except Exception:
                    raw_data = b64decode(part.data)

                yield {
                    "type": "input_audio",
                    "audio": b64encode(raw_data).decode(),
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
                "text": part.to_str(),
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
                "text": part.to_str(),
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

            response_output.append(part.to_str())

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
            "output": _tool_result(response.result),
        },
    )

    await connection.response.create()


def _prepare_session_config(
    *,
    instructions: ModelInstructions | None,
    config: OpenAIRealtimeConfig,
    tools: ModelTools,
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

    tool_choice: str | Mapping[str, str]
    match tools.selection:
        case "auto":
            tool_choice = "auto"

        case "required":
            tool_choice = "required"

        case "none":
            tool_choice = "none"

        case tool:
            tool_choice = {
                "type": "function",
                "name": tool.name,
            }

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
            for tool in tools.specification
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


def _event_context(
    event: ModelSessionEvent,
    /,
) -> ModelContext:
    if event.content is None:
        return ()

    if isinstance(event.content, State):
        return ()

    return event.content


def _content_to_multimodal(
    content: Any,
    /,
    *,
    audio_format: str,
) -> Generator[MultimodalContentPart]:
    for element in content:
        match element.get("type"):
            case "output_audio" | "input_audio" | "audio":
                if encoded_audio := element.get("audio"):
                    try:
                        yield ResourceContent.of(
                            b64decode(encoded_audio),
                            mime_type=audio_format,
                        )

                    except Exception as exc:
                        ctx.log_warning(
                            "Failed to decode audio content",
                            exception=exc,
                        )

            case "output_text" | "input_text":
                if text := element.get("text"):
                    yield TextContent.of(text)

                if transcript := element.get("transcript"):
                    yield TextContent.of(
                        transcript,
                        meta={"transcript": True},
                    )

            case other:
                ctx.log_warning(f"Unsupported message content - {other}")
