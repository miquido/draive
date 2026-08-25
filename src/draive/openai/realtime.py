import json
from base64 import b64decode
from collections.abc import Generator, Mapping, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from copy import copy
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Literal
from uuid import UUID, uuid4

from haiway import MISSING, Meta, Missing, State, ctx, without_missing
from openai.resources.realtime.realtime import (
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)
from openai.types.realtime import (
    RealtimeFunctionToolParam,
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
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus

from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputInvalid,
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
    def session_prepare(  # noqa: C901, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelSessionOutputSelection,
        config: OpenAIRealtimeConfig | None = None,
        **extra: Any,
    ) -> ModelSessionScope:
        assert isinstance(config, OpenAIRealtimeConfig | None)  # nosec: B101
        # managing scope manually
        scope: AbstractAsyncContextManager[str]
        # prepare config
        config = config or ctx.state(OpenAIRealtimeConfig)
        session_config: RealtimeSessionCreateRequestParam = _prepare_session_config(
            config=config,
            instructions=instructions,
            tools=tools,
            output=output,
        )
        output_audio_format: str
        match config.output_parameters:
            case {"format": {"type": str() as audio_output_type}}:
                output_audio_format = audio_output_type

            case _:
                output_audio_format = "audio/pcm"

        input_audio_format: str
        match config.input_parameters:
            case {"format": {"type": str() as audio_input_type}}:
                input_audio_format = audio_input_type

            case _:
                input_audio_format = "audio/pcm"
        # prepare connection
        connection_manager: AsyncRealtimeConnectionManager = self._client.realtime.connect(
            model=config.model,
            websocket_connection_options={
                "max_size": None,  # explicitly define no size limit
            },
        )

        async def open_session() -> ModelSession:  # noqa: C901, PLR0915
            nonlocal scope
            # enter scope
            scope = ctx.scope("model.session")
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
            # audio appended to the input buffer since its last commit
            buffered_audio: bool = False

            if context:  # send initial context
                await _send_context(
                    context,
                    current_items=current_items,
                    connection=connection,
                )

            async def read() -> ModelSessionOutputChunk:  # noqa: C901, PLR0911, PLR0912, PLR0915
                nonlocal current_items
                nonlocal buffered_audio
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
                                    # also emitted for interrupted, incomplete or cancelled
                                    if event.item.status != "completed":
                                        continue  # skip unfinished tool calls

                                    arguments: Mapping[str, Any] | None
                                    try:
                                        arguments = (
                                            json.loads(event.item.arguments)
                                            if event.item.arguments
                                            else None
                                        )

                                    except Exception as exc:
                                        raise ModelOutputInvalid(
                                            provider="openai",
                                            model=config.model,
                                            reason=(
                                                "Tool arguments decoding error - "
                                                f"{type(exc).__name__}: {exc}"
                                            ),
                                        ) from exc

                                    return ModelToolRequest.of(
                                        event.item.call_id or str(uuid4()),
                                        tool=event.item.name,
                                        arguments=arguments,
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
                            # the buffer was already consumed, nothing left to commit
                            buffered_audio = False
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

                            match event.response.status:
                                case "failed":
                                    raise ModelException(
                                        "Realtime response failed: "
                                        + _response_status_reason(event.response.status_details),
                                        provider="openai",
                                        model=config.model,
                                    )

                                case "incomplete":
                                    ctx.log_warning(
                                        "Realtime response incomplete: "
                                        + _response_status_reason(event.response.status_details)
                                    )

                                case _:
                                    pass  # nothing to report

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
                                                audio_format=input_audio_format,
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
                nonlocal buffered_audio
                if isinstance(part, ResourceContent) and part.mime_type.startswith("audio"):
                    # the input buffer accepts only audio matching the session input format
                    await connection.input_audio_buffer.append(audio=part.data)
                    buffered_audio = True
                    return

                # everything else has to be delivered as a conversation item,
                # the input buffer accepts audio only
                content_parts: Sequence[UserContentParam] = tuple(
                    _user_content_parts(MultimodalContent.of(part))
                )
                if not content_parts:
                    return  # nothing supported to send, already reported

                item_id: str = _item_identifier(part.meta.identifier)
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
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": content_parts,
                    },
                )

            async def write(
                input: ModelSessionInputChunk,  # noqa: A002
            ) -> None:
                nonlocal buffered_audio
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
                        if buffered_audio:
                            # the buffer can only be committed when it holds audio
                            await connection.input_audio_buffer.commit()
                            buffered_audio = False

                        # committing the input does not request the response on its own
                        await connection.response.create()

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
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            try:
                await connection_manager.__aexit__(  # close connection
                    exc_type,
                    exc_val,
                    exc_tb,
                )

            finally:
                await scope.__aexit__(  # exit scope
                    exc_type,
                    exc_val,
                    exc_tb,
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
                identifier: str = _item_identifier(input_element.meta.identifier)
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
                    item_id: str = _item_identifier(response.meta.identifier)
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
                            "output": _tool_result(response.content),
                        },
                    )

            case ModelOutput() as output_element:
                identifier: str = _item_identifier(output_element.meta.identifier)
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
                    item_id: str = _item_identifier(request.meta.identifier)
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


def _item_identifier(
    identifier: UUID | None,
    /,
) -> str:
    # the api limits conversation item identifiers to 32 characters,
    # the uuid text form takes 36 so the hex form is used instead
    return (identifier if identifier is not None else uuid4()).hex


def _user_content_parts(
    content: MultimodalContent,
) -> Generator[UserContentParam]:
    for part in content.parts:
        if isinstance(part, TextContent):
            # `transcript` is not sent to the model, `text` is the only field reaching it
            yield {
                "type": "input_text",
                "text": part.text,
            }

        elif isinstance(part, ResourceContent):
            if part.mime_type.startswith("audio"):
                yield {
                    "type": "input_audio",
                    # `data` already holds standard base64, which is what the API takes
                    "audio": part.data,
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
            "output": _tool_result(response.content),
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
    # the api allows only a single output modality per session
    modalities: list[Literal["text", "audio"]] = [_resolve_output_modality(output)]

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

    session_tools: list[RealtimeFunctionToolParam] | Missing
    if tools:
        session_tools = [
            without_missing(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or MISSING,
                    "parameters": tool.parameters or MISSING,
                },
                typed=RealtimeFunctionToolParam,
            )
            for tool in tools.specification
        ]

    else:
        session_tools = MISSING

    return without_missing(
        {
            "type": "realtime",
            "model": config.model,
            # an empty string would override the model's own default instructions
            "instructions": instructions or MISSING,
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


def _resolve_output_modality(
    output: ModelSessionOutputSelection,
    /,
) -> Literal["text", "audio"]:
    match output:
        case "auto" | "audio":
            return "audio"

        case "text":
            return "text"

        case output_selection if "audio" in output_selection:
            if len(output_selection) > 1:
                ctx.log_warning(
                    "OpenAI realtime supports a single output modality per session."
                    " Dropping unsupported output modalities and using audio."
                )

            return "audio"

        case output_selection if "text" in output_selection:
            if len(output_selection) > 1:
                ctx.log_warning(
                    "OpenAI realtime supports a single output modality per session."
                    " Dropping unsupported output modalities and using text."
                )

            return "text"

        case _:
            raise ValueError(f"Unsupported realtime output: {output}")


def _response_status_reason(
    status_details: RealtimeResponseStatus | None,
    /,
) -> str:
    if status_details is None:
        return "unknown"

    if error := status_details.error:
        return f"{error.type or 'error'} - {error.code or 'unknown'}"

    return status_details.reason or "unknown"


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
    content: Sequence[AssistantContent | UserContent],
    /,
    *,
    audio_format: str,
) -> Generator[MultimodalContentPart]:
    for element in content:
        match element.type:
            case "output_audio" | "input_audio":
                if encoded_audio := element.audio:
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

                # assistant audio resources can't be sent back within the context while their
                # transcript can - user audio is sent back as is, its transcript would duplicate
                if element.type == "output_audio" and (transcript := element.transcript):
                    yield TextContent.of(
                        transcript,
                        meta={"transcript": True},
                    )

            case "output_text" | "input_text":
                if text := element.text:
                    yield TextContent.of(text)

                if transcript := element.transcript:
                    yield TextContent.of(
                        transcript,
                        meta={"transcript": True},
                    )

            case other:
                ctx.log_warning(f"Unsupported message content - {other}")
