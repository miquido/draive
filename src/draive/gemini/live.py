from base64 import b64decode, b64encode, urlsafe_b64decode
from collections import deque
from collections.abc import AsyncIterator, Generator, MutableSequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import uuid4

from google.api_core.exceptions import ResourceExhausted  # pyright: ignore[reportMissingImport]
from google.genai.errors import ClientError
from google.genai.live import AsyncSession
from google.genai.types import (
    Blob,
    Content,
    ContentDict,
    FunctionDeclarationDict,
    LiveConnectConfigDict,
    LiveServerContent,
    LiveServerMessage,
    LiveServerSessionResumptionUpdate,
    MediaResolution,
    Modality,
    Part,
    PartDict,
    Transcription,
)
from haiway import MISSING, Meta, as_dict, ctx

from draive.gemini.api import GeminiAPI
from draive.gemini.config import GeminiConfig
from draive.gemini.utils import speech_config, unwrap_missing
from draive.models import (
    ModelContext,
    ModelException,
    ModelInput,
    ModelInputBlocks,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputBlocks,
    ModelOutputChunk,
    ModelReasoning,
    ModelReasoningChunk,
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
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference

__all__ = ("GeminiLive",)


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
                stop_sequences=config.stop_sequences,
                thinking_budget=config.thinking_budget,
            )

            session: AsyncSession = await connection_manager.__aenter__()

            if context:  # send initial context
                await session.send_client_content(
                    turns=_request_content(context),
                    turn_complete=False,
                )

            read_stream: AsyncIterator[LiveServerMessage] = session.receive()
            read_buffer: deque[ModelSessionOutputChunk] = deque()

            turn_input_transcript_parts: MutableSequence[TextContent] = []
            turn_output_transcript_parts: MutableSequence[TextContent] = []
            turn_output_blocks: MutableSequence[ModelOutputBlock] = []
            resumption: LiveServerSessionResumptionUpdate | None = None

            async def read() -> ModelSessionOutputChunk:  # noqa: C901, PLR0912, PLR0915
                nonlocal read_buffer
                nonlocal read_stream
                nonlocal turn_input_transcript_parts
                nonlocal turn_output_transcript_parts
                nonlocal turn_output_blocks
                nonlocal resumption

                while True:
                    if read_buffer:
                        return read_buffer.popleft()

                    try:
                        message: LiveServerMessage = await anext(read_stream)

                    except ResourceExhausted as exc:
                        raise ModelException(
                            "Gemini Live session rate limited",
                            provider="gemini",
                            model=config.model,
                        ) from exc

                    except ClientError as exc:
                        if exc.code == 429:  # noqa: PLR2004
                            raise ModelException(
                                "Gemini Live session rate limited",
                                provider="gemini",
                                model=config.model,
                            ) from exc

                        raise ModelException(
                            str(exc),
                            provider="gemini",
                            model=config.model,
                        ) from exc

                    except StopAsyncIteration:
                        read_stream = session.receive()
                        continue  # for some mysterious reason iterator breaks after each turn...

                    if message.usage_metadata is not None:
                        record_usage_metrics(
                            provider="gemini",
                            model=config.model,
                            input_tokens=message.usage_metadata.prompt_token_count,
                            cached_input_tokens=message.usage_metadata.cached_content_token_count,
                            output_tokens=message.usage_metadata.response_token_count,
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
                                for chunk in _part_as_stream_elements(part):
                                    read_buffer.append(chunk)

                                turn_output_blocks.extend(_part_as_output_blocks(part))

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
                                                "reason": (
                                                    server_content.turn_complete_reason
                                                    if server_content.turn_complete_reason
                                                    else None
                                                ),
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

                    if message.session_resumption_update is not None:
                        resumption = message.session_resumption_update

                    if message.go_away is not None:
                        # TODO: support automatic reconnect/session resumption.
                        raise NotImplementedError()

            async def write(  # noqa: C901
                input: ModelSessionInputChunk,  # noqa: A002
            ) -> None:
                if isinstance(input, ResourceContent):
                    if input.mime_type.startswith("audio"):
                        await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                            audio=Blob(
                                data=input.to_bytes(),
                                mime_type=input.mime_type,
                            )
                        )

                    elif input.mime_type.startswith("image"):
                        await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                            media=Blob(
                                data=input.to_bytes(),
                                mime_type=input.mime_type,
                            ),
                        )

                    elif input.mime_type.startswith("video"):
                        await session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                            video=Blob(
                                data=input.to_bytes(),
                                mime_type=input.mime_type,
                            ),
                        )

                    else:
                        ctx.log_error(
                            "Gemini Live input media "
                            f"({input.mime_type}) not supported! Skipping..."
                        )

                elif isinstance(input, ModelToolResponse):
                    await session.send_tool_response(
                        function_responses=[
                            {
                                "id": input.identifier,
                                "name": input.tool,
                                "response": _tool_response_payload(input),
                            }
                        ]
                    )

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
                await scope.__aexit__(  # noqa: F821
                    exc_type,
                    exc_val,
                    exc_tb,
                )

        return ModelSessionScope(
            opening=open_session,
            closing=close_session,
        )


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
        "context_window_compression": {
            "sliding_window": {},
        },
        "seed": unwrap_missing(config.seed),
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

    if config.media_resolution is MISSING:
        pass  # skip missing

    elif config.media_resolution == "low":
        live_config["media_resolution"] = MediaResolution.MEDIA_RESOLUTION_LOW

    elif config.media_resolution == "medium":
        live_config["media_resolution"] = MediaResolution.MEDIA_RESOLUTION_MEDIUM

    elif config.media_resolution == "high":
        live_config["media_resolution"] = MediaResolution.MEDIA_RESOLUTION_HIGH

    else:
        raise ValueError(f"Unsupported media resolution: {config.media_resolution}")

    if speech := speech_config(config):
        live_config["speech_config"] = speech

    if config.thinking_budget is not MISSING:
        live_config["thinking_config"] = {
            "include_thoughts": True,
            "thinking_budget": cast(int, config.thinking_budget),
        }

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


def _tool_response_payload(
    response: ModelToolResponse,
    /,
) -> dict[str, Any]:
    # TODO: add media support within tool responses
    if response.status == "error":
        return {"error": "".join(part.to_str() for part in response.result.parts)}

    else:
        return {"output": "".join(part.to_str() for part in response.result.parts)}


def _request_content(
    context: ModelContext,
) -> list[Content | ContentDict]:
    content: list[Content | ContentDict] = []
    for element in context:
        if isinstance(element, ModelInput):
            content.append(
                ContentDict(
                    role="user",
                    parts=list(_block_parts(element.input)),
                )
            )

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            content.append(
                ContentDict(
                    role="model",
                    parts=list(_block_parts(element.output)),
                )
            )

    return content


def _part_as_stream_elements(
    part: Part,
) -> Generator[ModelOutputChunk]:
    if part.text:
        if part.thought:
            yield ModelReasoningChunk.of(
                TextContent.of(part.text),
                meta={
                    "kind": "thought",
                    "signature": b64encode(part.thought_signature).decode()
                    if part.thought_signature
                    else None,
                },
            )

        else:
            yield TextContent.of(part.text)

    if part.function_call and part.function_call.name:
        yield ModelToolRequest(
            identifier=part.function_call.id or str(uuid4()),
            tool=part.function_call.name,
            arguments=part.function_call.args if part.function_call.args is not None else {},
            meta=Meta.of(
                {
                    "signature": b64encode(part.thought_signature).decode(),
                }
            )
            if part.thought_signature
            else Meta.empty,
        )

    if part.inline_data and part.inline_data.data:
        yield ResourceContent.of(
            part.inline_data.data,
            mime_type=part.inline_data.mime_type or "application/octet-stream",
        )

    if part.file_data and part.file_data.file_uri:
        yield ResourceReference.of(
            part.file_data.file_uri,
            mime_type=part.file_data.mime_type,
        )


def _part_as_output_blocks(
    part: Part,
) -> Generator[ModelOutputBlock]:
    if part.text:
        if part.thought:
            yield ModelReasoning.of(
                TextContent.of(part.text),
                meta={
                    "kind": "thought",
                    "signature": b64encode(part.thought_signature).decode()
                    if part.thought_signature
                    else None,
                },
            )

        else:
            yield MultimodalContent.of(TextContent.of(part.text))

    if part.function_call and part.function_call.name:
        yield ModelToolRequest.of(
            part.function_call.id or str(uuid4()),
            tool=part.function_call.name,
            arguments=part.function_call.args if part.function_call.args is not None else {},
            meta=Meta.of(
                {
                    "signature": b64encode(part.thought_signature).decode(),
                }
            )
            if part.thought_signature
            else Meta.empty,
        )

    if part.inline_data and part.inline_data.data:
        yield MultimodalContent.of(
            ResourceContent.of(
                part.inline_data.data,
                mime_type=part.inline_data.mime_type or "application/octet-stream",
            )
        )

    if part.file_data and part.file_data.file_uri:
        yield MultimodalContent.of(
            ResourceReference.of(
                part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )


def _block_parts(
    blocks: ModelInputBlocks | ModelOutputBlocks,
    /,
) -> Generator[PartDict]:
    for block in blocks:
        if isinstance(block, ModelToolRequest):
            if signature := block.meta.get_str("signature"):
                yield {
                    "function_call": {
                        "id": block.identifier,
                        "name": block.tool,
                        "args": as_dict(block.arguments),
                    },
                    "thought_signature": b64decode(signature),
                }

            else:
                yield {
                    "function_call": {
                        "id": block.identifier,
                        "name": block.tool,
                        "args": as_dict(block.arguments),
                    }
                }

        elif isinstance(block, ModelToolResponse):
            yield {
                "function_response": {
                    "id": block.identifier,
                    "name": block.tool,
                    "response": _tool_response_payload(block),
                }
            }

        elif isinstance(block, ModelReasoning):
            if block.meta.kind == "thought":
                if signature := block.meta.get_str("signature"):
                    yield {
                        "text": block.reasoning.to_str(),
                        "thought": True,
                        "thought_signature": b64decode(signature),
                    }

                else:
                    yield {
                        "text": block.reasoning.to_str(),
                        "thought": True,
                    }

            else:
                raise ValueError(f"Unsupported reasoning element: {block.meta.kind}")

        else:
            assert isinstance(block, MultimodalContent)  # nosec: B101
            yield from _content_parts(block)


def _content_parts(
    content: MultimodalContent,
    /,
) -> Generator[PartDict]:
    for part in content.parts:
        if isinstance(part, TextContent):
            yield {"text": part.text}

        elif isinstance(part, ResourceContent):
            yield {
                "inline_data": {
                    "data": urlsafe_b64decode(part.data),
                    "mime_type": part.mime_type,
                }
            }

        elif isinstance(part, ResourceReference):
            yield {
                "file_data": {
                    "file_uri": part.uri,
                    "mime_type": part.mime_type,
                }
            }

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            if part.hidden:
                continue

            yield {"text": part.to_str()}
