from base64 import b64encode
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

import pytest
from haiway import Meta, ctx
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content

from draive.models import (
    ModelInput,
    ModelOutput,
    ModelSessionEvent,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.openai.config import OpenAIRealtimeConfig
from draive.openai.realtime import (
    OpenAIRealtime,
    _assistant_content_parts,
    _content_to_multimodal,
    _item_identifier,
    _prepare_session_config,
    _reset_context,
    _send_context,
    _user_content_parts,
)
from draive.resources import ResourceContent


def test_content_to_multimodal_decodes_input_audio() -> None:
    raw_audio = b"\x00\x01\x02\x03"
    encoded_audio = b64encode(raw_audio).decode("ascii")

    content = MultimodalContent.of(
        *_content_to_multimodal(
            [
                Content(
                    type="input_audio",
                    audio=encoded_audio,
                )
            ],
            audio_format="audio/pcm",
        )
    )

    assert len(content.parts) == 1
    part = content.parts[0]
    assert isinstance(part, ResourceContent)
    assert part.mime_type == "audio/pcm"
    assert part.to_bytes() == raw_audio


class _MockConversationItem:
    def __init__(self) -> None:
        self.created_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.created_items: list[Mapping[str, Any]] = []

    async def create(
        self,
        *,
        item: Mapping[str, Any],
    ) -> None:
        self.created_items.append(item)
        item_id = item.get("id")
        if isinstance(item_id, str):
            self.created_ids.append(item_id)

    async def delete(
        self,
        *,
        item_id: str,
    ) -> None:
        self.deleted_ids.append(item_id)


class _MockConversation:
    def __init__(self) -> None:
        self.item = _MockConversationItem()


class _MockResponse:
    def __init__(self) -> None:
        self.create_calls: int = 0

    async def create(self) -> None:
        self.create_calls += 1


class _MockInputAudioBuffer:
    def __init__(self) -> None:
        self.appended_audio: list[str] = []
        self.commit_calls: int = 0

    async def append(
        self,
        *,
        audio: str,
    ) -> None:
        self.appended_audio.append(audio)

    async def commit(self) -> None:
        self.commit_calls += 1


class _MockConnection:
    def __init__(self) -> None:
        self.conversation = _MockConversation()
        self.response = _MockResponse()
        self.input_audio_buffer = _MockInputAudioBuffer()
        self._events: list[Any] = []

    async def recv(self) -> Any:
        if self._events:
            return self._events.pop(0)

        raise RuntimeError("No events queued for _MockConnection.recv()")


class _MockConnectionManager:
    def __init__(self, connection: _MockConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _MockConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        return None


class _MockSessionResource:
    def __init__(self) -> None:
        self.updates: list[Mapping[str, Any]] = []

    async def update(
        self,
        *,
        session: Mapping[str, Any],
    ) -> None:
        self.updates.append(session)


class _MockRealtimeConnect:
    def __init__(self, manager: _MockConnectionManager) -> None:
        self._manager = manager

    def connect(
        self,
        *,
        model: str,
        websocket_connection_options: Mapping[str, Any],
    ) -> _MockConnectionManager:
        return self._manager


class _MockClient:
    def __init__(
        self,
        connection_manager: _MockConnectionManager,
    ) -> None:
        self.realtime = _MockRealtimeConnect(connection_manager)


def _realtime_config() -> OpenAIRealtimeConfig:
    return OpenAIRealtimeConfig(
        model="gpt-realtime",
        input_parameters={"format": {"type": "audio/pcm", "rate": 24000}},
        output_parameters={"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"},
    )


async def _prepared_session(connection: _MockConnection) -> Any:
    connection.session = _MockSessionResource()
    manager = _MockConnectionManager(connection)
    model = OpenAIRealtime(api_key="test")
    model._client = cast(Any, _MockClient(manager))

    return model.session_prepare(
        instructions="",
        tools=ModelTools.none,
        context=(),
        output="text",
        config=_realtime_config(),
    )


@pytest.mark.asyncio
async def test_session_write_text_content_creates_conversation_item() -> None:
    connection = _MockConnection()
    scope = await _prepared_session(connection)

    session = await scope.__aenter__()
    await session._writing(TextContent.of("hello"))

    # the input buffer accepts audio only, text has to become a conversation item
    assert connection.input_audio_buffer.appended_audio == []
    assert len(connection.conversation.item.created_items) == 1
    item = connection.conversation.item.created_items[0]
    assert item["role"] == "user"
    assert item["content"] == ({"type": "input_text", "text": "hello"},)
    # the item alone does not request a response
    assert connection.response.create_calls == 0

    await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_session_turn_commit_requests_response_without_buffered_audio() -> None:
    connection = _MockConnection()
    scope = await _prepared_session(connection)

    session = await scope.__aenter__()
    await session._writing(TextContent.of("hello"))
    await session._writing(ModelSessionEvent.turn_commited())

    # committing an empty input buffer is rejected by the api
    assert connection.input_audio_buffer.commit_calls == 0
    # committing does not request the response on its own
    assert connection.response.create_calls == 1

    await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_session_turn_commit_commits_buffered_audio_once() -> None:
    connection = _MockConnection()
    scope = await _prepared_session(connection)

    session = await scope.__aenter__()
    await session._writing(ResourceContent.of(b"\x00\x01", mime_type="audio/pcm"))
    await session._writing(ModelSessionEvent.turn_commited())

    assert connection.input_audio_buffer.commit_calls == 1
    assert connection.response.create_calls == 1

    # the buffer was consumed, a subsequent turn only requests the response
    await session._writing(ModelSessionEvent.turn_commited())

    assert connection.input_audio_buffer.commit_calls == 1
    assert connection.response.create_calls == 2

    await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_session_write_hidden_artifact_is_skipped() -> None:
    connection = _MockConnection()
    scope = await _prepared_session(connection)

    session = await scope.__aenter__()
    await session._writing(
        ArtifactContent.of(
            {"value": "artifact body"},
            category="note",
            hidden=True,
        )
    )

    assert connection.response.create_calls == 0
    assert connection.conversation.item.created_items == []

    await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_reset_context_deletes_tool_items_seeded_by_send_context() -> None:
    connection = _MockConnection()
    tool_response = ModelToolResponse.of(
        "tool-call-output",
        tool="echo",
        content=MultimodalContent.of(TextContent.of("ok")),
        meta={"identifier": "00000000-0000-0000-0000-000000000013"},
    )
    tool_request = ModelToolRequest.of(
        "tool-call-request",
        tool="echo",
        arguments={},
        meta={"identifier": "00000000-0000-0000-0000-000000000014"},
    )
    context = (
        ModelInput.of(
            MultimodalContent.of(TextContent.of("user")),
            tool_response,
            meta={"identifier": "00000000-0000-0000-0000-000000000011"},
        ),
        ModelOutput.of(
            MultimodalContent.of(TextContent.of("assistant")),
            tool_request,
            meta={"identifier": "00000000-0000-0000-0000-000000000012"},
        ),
    )
    current_items: dict[str, Meta] = {}

    await _send_context(
        context,
        current_items=current_items,
        connection=cast(Any, connection),
    )

    # identifiers are shortened to the hex form accepted by the api
    assert set(current_items.keys()) == {
        "00000000000000000000000000000011",
        "00000000000000000000000000000012",
        "00000000000000000000000000000013",
        "00000000000000000000000000000014",
    }

    await _reset_context(
        (),
        current_items=current_items,
        connection=cast(Any, connection),
    )

    assert set(connection.conversation.item.deleted_ids) == {
        "00000000000000000000000000000011",
        "00000000000000000000000000000012",
        "00000000000000000000000000000013",
        "00000000000000000000000000000014",
    }
    assert current_items == {}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param("auto", ["audio"], id="auto"),
        pytest.param("text", ["text"], id="text"),
        pytest.param("audio", ["audio"], id="audio"),
        pytest.param(["audio"], ["audio"], id="audio-list"),
        pytest.param(("text",), ["text"], id="text-tuple"),
        pytest.param({"audio"}, ["audio"], id="audio-set"),
        pytest.param(["text", "audio"], ["audio"], id="both-narrowed-to-audio"),
        pytest.param(["audio", "text"], ["audio"], id="both-reversed-narrowed-to-audio"),
    ],
)
@pytest.mark.asyncio
async def test_session_config_resolves_a_single_output_modality(
    output: Any,
    expected: list[str],
) -> None:
    # the api allows only one output modality - any collection has to be narrowed to one
    async with ctx.scope("test"):
        session_config = _prepare_session_config(
            instructions="",
            config=_realtime_config(),
            tools=ModelTools.none,
            output=output,
        )

    assert session_config["output_modalities"] == expected


def test_user_content_parts_send_a_transcript_as_text() -> None:
    # the `transcript` field is not sent to the model, only `text` reaches it
    parts = list(
        _user_content_parts(
            MultimodalContent.of(
                TextContent.of(
                    "spoken words",
                    meta={"transcript": True},
                )
            )
        )
    )

    assert parts == [{"type": "input_text", "text": "spoken words"}]


def test_assistant_audio_round_trips_through_its_transcript() -> None:
    # audio resources can't be sent back, the transcript is the only representation left
    content = MultimodalContent.of(
        *_content_to_multimodal(
            [
                AssistantContent(
                    type="output_audio",
                    audio=b64encode(b"\x00\x01").decode("ascii"),
                    transcript="spoken response",
                )
            ],
            audio_format="audio/pcm",
        )
    )

    assert any(isinstance(part, ResourceContent) for part in content.parts)
    assert list(_assistant_content_parts(content)) == [
        {"type": "output_text", "text": "spoken response"}
    ]


@pytest.mark.asyncio
async def test_session_write_image_resource_is_not_appended_as_audio() -> None:
    connection = _MockConnection()
    connection.session = _MockSessionResource()
    manager = _MockConnectionManager(connection)
    model = OpenAIRealtime(api_key="test")
    model._client = cast(Any, _MockClient(manager))

    scope = model.session_prepare(
        instructions="",
        tools=ModelTools.none,
        context=(),
        output="audio",
        config=_realtime_config(),
    )

    session = await scope.__aenter__()
    await session._writing(
        ResourceContent.of(
            b"\x89PNG",
            mime_type="image/png",
        )
    )

    # only audio matching the session input format may reach the input buffer
    assert connection.input_audio_buffer.appended_audio == []

    await scope.__aexit__(None, None, None)


def test_user_audio_does_not_duplicate_its_transcript() -> None:
    # user audio is sent back as is - adding its transcript would duplicate the content
    content = MultimodalContent.of(
        *_content_to_multimodal(
            [
                Content(
                    type="input_audio",
                    audio=b64encode(b"\x00\x01").decode("ascii"),
                    transcript="spoken words",
                )
            ],
            audio_format="audio/pcm",
        )
    )

    assert len(content.parts) == 1
    assert isinstance(content.parts[0], ResourceContent)


def test_item_identifier_fits_api_length_limit() -> None:
    identifier = UUID("00000000-0000-0000-0000-000000000011")

    # the api rejects conversation item identifiers longer than 32 characters
    assert _item_identifier(identifier) == "00000000000000000000000000000011"
    assert len(_item_identifier(None)) == 32


@pytest.mark.asyncio
async def test_send_context_uses_short_item_identifiers() -> None:
    connection = _MockConnection()
    await _send_context(
        (
            ModelInput.of(
                MultimodalContent.of(TextContent.of("user")),
                meta={"identifier": "00000000-0000-0000-0000-000000000011"},
            ),
            ModelOutput.of(
                MultimodalContent.of(TextContent.of("assistant")),
                ModelToolRequest.of(
                    "tool-call-request",
                    tool="echo",
                    arguments={},
                    meta={"identifier": "00000000-0000-0000-0000-000000000013"},
                ),
                meta={"identifier": "00000000-0000-0000-0000-000000000012"},
            ),
        ),
        current_items={},
        connection=cast(Any, connection),
    )

    assert connection.conversation.item.created_ids
    assert all(len(item_id) <= 32 for item_id in connection.conversation.item.created_ids)
