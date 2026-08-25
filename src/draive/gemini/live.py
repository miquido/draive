from collections import deque
from collections.abc import AsyncIterator, MutableSequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Final
from uuid import uuid4

from google.genai.errors import APIError
from google.genai.live import AsyncSession
from google.genai.types import (
    Blob,
    Content,
    ContentDict,
    FunctionDeclarationDict,
    LiveConnectConfigDict,
    LiveServerContent,
    LiveServerMessage,
    MediaResolution,
    Modality,
    Transcription,
    TurnCompleteReason,
)
from haiway import MISSING, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiConfig
from draive.gemini.content import (
    block_parts,
    function_response,
    part_as_output_blocks,
    part_as_stream_elements,
)
from draive.gemini.utils import (
    RATE_LIMIT_STATUS_CODE,
    combined_input_tokens,
    speech_config,
    thinking_config,
    unwrap_missing,
)
from draive.models import (
    ModelContext,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputFailed,
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
from draive.models.metrics import (
    model_rate_limit,
    record_model_invocation,
    record_usage_metrics,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent

__all__ = ("GeminiLive",)

# turn completion reasons which indicate a failure instead of a regular completion
_TURN_FAILURE_REASONS: Final[frozenset[TurnCompleteReason]] = frozenset(
    (
        TurnCompleteReason.MALFORMED_FUNCTION_CALL,
        TurnCompleteReason.RESPONSE_REJECTED,
        TurnCompleteReason.PROHIBITED_INPUT_CONTENT,
        TurnCompleteReason.IMAGE_PROHIBITED_INPUT_CONTENT,
        TurnCompleteReason.BLOCKLIST,
        TurnCompleteReason.GENERATED_CONTENT_SAFETY,
        TurnCompleteReason.GENERATED_IMAGE_SAFETY,
        TurnCompleteReason.GENERATED_AUDIO_SAFETY,
        TurnCompleteReason.GENERATED_VIDEO_SAFETY,
        TurnCompleteReason.GENERATED_CONTENT_PROHIBITED,
        TurnCompleteReason.GENERATED_CONTENT_BLOCKLIST,
    )
)


class GeminiLive(GeminiAPI):
    def session_prepare(  # noqa: C901, PLR0915
        self,
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: ModelContext,
        output: ModelSessionOutputSelection,
        config: GeminiConfig | None = None,
        **extra: Any,
    ) -> ModelSessionScope:
        assert isinstance(config, GeminiConfig | None)  # nosec: B101
        # managing scope manually
        scope: AbstractAsyncContextManager[str]
        # prepare config
        config = config or ctx.state(GeminiConfig)
        connection_config: LiveConnectConfigDict = _live_connect_config(
            instructions=instructions,
            tools=tools,
            output=output,
            config=config,
        )
        # prepare connection
        connection_manager: AbstractAsyncContextManager[AsyncSession] = (
            self._client.aio.live.connect(
                model=config.model,
                config=connection_config,
            )
        )

        async def open_session() -> ModelSession:  # noqa: C901, PLR0915
            nonlocal scope
            # enter scope
            scope = ctx.scope("model.session")
            await scope.__aenter__()
            record_model_invocation(
                provider="gemini",
                model=config.model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                tools=tools,
                output=output,
                # stop sequences are not supported within live sessions
                top_p=config.top_p,
                top_k=config.top_k,
                seed=config.seed,
                thinking_budget=config.thinking_budget,
                thinking_level=config.thinking_level,
            )

            session: AsyncSession = await connection_manager.__aenter__()

            # seed the session with prior context as an incomplete turn - the api rejects
            # a completed client content turn, realtime input starts the turn instead
            if context:
                await session.send_client_content(
                    turns=_request_content(context),
                    turn_complete=False,
                )

            read_stream: AsyncIterator[LiveServerMessage] = session.receive()
            read_buffer: deque[ModelSessionOutputChunk] = deque()

            turn_input_transcript_parts: MutableSequence[TextContent] = []
            turn_output_transcript_parts: MutableSequence[TextContent] = []
            turn_output_blocks: MutableSequence[ModelOutputBlock] = []

            async def read() -> ModelSessionOutputChunk:  # noqa: C901, PLR0912, PLR0915
                nonlocal read_buffer
                nonlocal read_stream
                nonlocal turn_input_transcript_parts
                nonlocal turn_output_transcript_parts
                nonlocal turn_output_blocks

                while True:
                    if read_buffer:
                        return read_buffer.popleft()

                    try:
                        message: LiveServerMessage = await anext(read_stream)

                    except APIError as exc:
                        # the Live API reports failures through websocket close codes,
                        # so a rate limit is only recognizable from the close reason
                        exc_details: str = str(exc)
                        if (
                            exc.code == RATE_LIMIT_STATUS_CODE
                            or "RESOURCE_EXHAUSTED" in exc_details
                            or "quota" in exc_details.lower()
                        ):
                            raise model_rate_limit(
                                provider="gemini",
                                model=config.model,
                                retry_after=None,
                            ) from exc

                        raise ModelOutputFailed(
                            provider="gemini",
                            model=config.model,
                            reason=str(exc),
                        ) from exc

                    except StopAsyncIteration:
                        read_stream = session.receive()
                        continue  # for some mysterious reason iterator breaks after each turn...

                    if message.usage_metadata is not None:
                        record_usage_metrics(
                            provider="gemini",
                            model=config.model,
                            input_tokens=combined_input_tokens(
                                message.usage_metadata.prompt_token_count,
                                message.usage_metadata.tool_use_prompt_token_count,
                            ),
                            cached_input_tokens=message.usage_metadata.cached_content_token_count,
                            output_tokens=message.usage_metadata.response_token_count,
                            # thinking tokens are not included within response count
                            reasoning_output_tokens=message.usage_metadata.thoughts_token_count,
                        )

                    if message.server_content is not None:
                        server_content: LiveServerContent = message.server_content
                        if server_content.input_transcription is not None:
                            transcription: Transcription = server_content.input_transcription
                            if transcription.text:
                                turn_input_transcript_parts.append(
                                    TextContent.of(
                                        transcription.text,
                                        meta={"transcript": True},
                                    )
                                )

                            if transcription.finished and turn_input_transcript_parts:
                                read_buffer.append(
                                    ModelSessionEvent.turn_completed(
                                        ModelInput.of(
                                            MultimodalContent.of(*turn_input_transcript_parts),
                                            meta={
                                                "created": datetime.now(UTC).isoformat(),
                                            },
                                        ),
                                        meta={
                                            "created": datetime.now(UTC).isoformat(),
                                        },
                                    )
                                )
                                turn_input_transcript_parts.clear()

                        if server_content.output_transcription is not None:
                            transcription = server_content.output_transcription
                            if transcription.text:
                                turn_output_transcript_parts.append(
                                    TextContent.of(
                                        transcription.text,
                                        meta={"transcript": True},
                                    )
                                )

                        if (
                            server_content.model_turn is not None
                            and server_content.model_turn.parts
                        ):
                            for part in server_content.model_turn.parts:
                                for chunk in part_as_stream_elements(part):
                                    read_buffer.append(chunk)

                                turn_output_blocks.extend(part_as_output_blocks(part))

                        if server_content.interrupted:
                            read_buffer.append(
                                ModelSessionEvent.turn_interrupted(
                                    meta={"created": datetime.now(UTC).isoformat()},
                                )
                            )

                        if server_content.generation_complete:
                            read_buffer.append(
                                ModelSessionEvent.turn_finished(
                                    meta={"created": datetime.now(UTC).isoformat()},
                                )
                            )

                        if server_content.turn_complete:
                            completion_reason: str | None = (
                                server_content.turn_complete_reason.value
                                if server_content.turn_complete_reason
                                else None
                            )
                            if server_content.turn_complete_reason in _TURN_FAILURE_REASONS:
                                ctx.log_warning(
                                    "Gemini Live turn completed with a failure"
                                    f" ({completion_reason})"
                                )

                            if turn_output_transcript_parts:
                                turn_output_blocks.append(
                                    MultimodalContent.of(*turn_output_transcript_parts)
                                )

                            if turn_output_blocks:
                                read_buffer.append(
                                    ModelSessionEvent.turn_completed(
                                        ModelOutput.of(
                                            *turn_output_blocks,
                                            meta={
                                                "created": datetime.now(UTC).isoformat(),
                                                "reason": completion_reason,
                                            },
                                        ),
                                        meta={
                                            "created": datetime.now(UTC).isoformat(),
                                            "interrupted": server_content.interrupted,
                                        },
                                    )
                                )

                                turn_output_blocks.clear()
                                turn_output_transcript_parts.clear()

                    if (
                        message.tool_call is not None
                        and message.tool_call.function_calls is not None
                    ):
                        for function_call in message.tool_call.function_calls:
                            if function_call.name is None:
                                continue

                            request: ModelToolRequest = ModelToolRequest.of(
                                function_call.id or str(uuid4()),
                                tool=function_call.name,
                                arguments=function_call.args,
                                meta={
                                    "identifier": function_call.id,
                                    "created": datetime.now(UTC).isoformat(),
                                    "will_continue": function_call.will_continue,
                                },
                            )
                            read_buffer.append(request)
                            turn_output_blocks.append(request)

                    if message.tool_call_cancellation is not None:
                        ctx.log_warning(
                            "Received unsupported tool cancellation event - skipping..."
                        )

                    if message.voice_activity is not None:
                        match message.voice_activity.voice_activity_type:
                            case "ACTIVITY_START":
                                read_buffer.append(
                                    ModelSessionEvent.turn_started(
                                        meta={"created": datetime.now(UTC).isoformat()},
                                    )
                                )

                            case "ACTIVITY_END":
                                read_buffer.append(
                                    ModelSessionEvent.turn_commited(
                                        meta={"created": datetime.now(UTC).isoformat()},
                                    )
                                )

                            case _:
                                pass

                    if message.go_away is not None:
                        # routine advance notice of the connection being closed,
                        # the actual closing surfaces through the stream itself
                        # TODO: support automatic reconnect/session resumption.
                        ctx.log_warning(
                            "Gemini Live session will be closed soon"
                            f" (time left: {message.go_away.time_left})"
                        )

            async def write_media(
                media: ResourceContent,
                /,
            ) -> None:
                # each modality has its own keyword within a single realtime input
                blob: Blob = Blob(
                    data=media.to_bytes(),
                    mime_type=media.mime_type,
                )
                if media.mime_type.startswith("audio"):
                    await session.send_realtime_input(audio=blob)  # pyright: ignore[reportUnknownMemberType]

                elif media.mime_type.startswith("image"):
                    await session.send_realtime_input(media=blob)  # pyright: ignore[reportUnknownMemberType]

                elif media.mime_type.startswith("video"):
                    await session.send_realtime_input(video=blob)  # pyright: ignore[reportUnknownMemberType]

                else:
                    ctx.log_warning(
                        f"Gemini Live input media ({media.mime_type}) not supported! Skipping..."
                    )

            async def write(
                input: ModelSessionInputChunk,  # noqa: A002
            ) -> None:
                if isinstance(input, ResourceContent):
                    await write_media(input)

                elif isinstance(input, ModelToolResponse):
                    await session.send_tool_response(function_responses=[function_response(input)])

                elif isinstance(input, TextContent):
                    await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                        text=input.text,
                    )

                elif isinstance(input, ModelSessionEvent):
                    match input.event:
                        case "turn_commited":
                            await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                                activity_end={},
                            )

                        case "turn_started":
                            await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                                activity_start={},
                            )

                        case "context_updated":
                            ctx.log_error(
                                "Gemini Live session context reset is not supported in-place."
                                " TODO: restart the session with refreshed context if we decide"
                                " to support provider-specific session rebuilds."
                            )

                        case _:
                            ctx.log_debug(f"Received unsupported input event: {input.event}")

                else:
                    ctx.log_warning(
                        f"Gemini Live input ({type(input).__name__}) not supported! Skipping..."
                    )

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
                await connection_manager.__aexit__(
                    exc_type,
                    exc_val,
                    exc_tb,
                )

            finally:
                await scope.__aexit__(
                    exc_type,
                    exc_val,
                    exc_tb,
                )

        return ModelSessionScope(
            opening=open_session,
            closing=close_session,
        )


def _resolve_media_resolution(config: GeminiConfig) -> MediaResolution | None:
    if config.media_resolution is MISSING:
        return None
    elif config.media_resolution == "low":
        return MediaResolution.MEDIA_RESOLUTION_LOW
    elif config.media_resolution == "medium":
        return MediaResolution.MEDIA_RESOLUTION_MEDIUM
    elif config.media_resolution == "high":
        return MediaResolution.MEDIA_RESOLUTION_HIGH
    else:
        raise ValueError(f"Unsupported media resolution: {config.media_resolution}")


def _live_connect_config(
    *,
    instructions: ModelInstructions,
    tools: ModelTools,
    output: ModelSessionOutputSelection,
    config: GeminiConfig,
) -> LiveConnectConfigDict:
    live_config: LiveConnectConfigDict = {
        "temperature": unwrap_missing(config.temperature),
        "top_p": unwrap_missing(config.top_p),
        "top_k": unwrap_missing(config.top_k),
        "max_output_tokens": unwrap_missing(config.max_output_tokens),
        "history_config": {
            "initial_history_in_client_content": True,
        },
        "seed": unwrap_missing(config.seed),
    }

    if config.context_window_compression is True:
        live_config["context_window_compression"] = {"sliding_window": {}}

    # activity_start/activity_end can only be sent when server side detection is off
    if config.automatic_activity_detection is False:
        live_config["realtime_input_config"] = {
            "automatic_activity_detection": {"disabled": True},
        }

    if instructions:
        live_config["system_instruction"] = instructions

    response_modality: Modality = _resolve_response_modality(output)
    live_config["response_modalities"] = [response_modality]
    if response_modality == Modality.AUDIO:
        live_config["output_audio_transcription"] = {}

    live_config["input_audio_transcription"] = {}

    if tools.specification:
        live_config["tools"] = [
            {
                "function_declarations": [
                    FunctionDeclarationDict(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.parameters,
                    )
                    for tool in tools.specification
                ]
            }
        ]

    if resolution := _resolve_media_resolution(config):
        live_config["media_resolution"] = resolution

    if speech := speech_config(config):
        live_config["speech_config"] = speech

    if thinking := thinking_config(config):
        live_config["thinking_config"] = thinking

    return live_config


def _resolve_response_modality(
    output: ModelSessionOutputSelection,
    /,
) -> Modality:
    match output:
        case "auto" | "audio":
            return Modality.AUDIO

        case "text":
            return Modality.TEXT

        case output_selection if "audio" in output_selection:
            ctx.log_warning(
                "Gemini Live supports a single response modality per session."
                " Dropping unsupported output modalities and using audio."
            )
            return Modality.AUDIO

        case output_selection if "text" in output_selection:
            ctx.log_warning(
                "Gemini Live supports a single response modality per session."
                " Dropping unsupported output modalities and using text."
            )
            return Modality.TEXT

        case _:
            raise ValueError(f"Unsupported realtime output: {output}")


def _request_content(
    context: ModelContext,
) -> list[Content | ContentDict]:
    content: list[Content | ContentDict] = []
    for element in context:
        if isinstance(element, ModelInput):
            content.append(
                ContentDict(
                    role="user",
                    parts=list(block_parts(element.input)),
                )
            )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            content.append(
                ContentDict(
                    role="model",
                    parts=list(block_parts(element.output)),
                )
            )

    return content
