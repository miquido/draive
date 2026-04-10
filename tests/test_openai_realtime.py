from base64 import b64encode
from collections.abc import Mapping
from typing import Any, cast

import pytest
from haiway import Meta
from openai.types.realtime.realtime_conversation_item_user_message import Content

from draive.models import (
    ModelInput,
    ModelOutput,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.openai.config import OpenAIRealtimeConfig
from draive.openai.realtime import (
    OpenAIRealtime,
    _content_to_multimodal,
    _reset_context,
    _send_context,
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


@pytest.mark.asyncio
async def test_session_write_text_content_is_skipped() -> None:
    connection = _MockConnection()
    connection.session = _MockSessionResource()
    manager = _MockConnectionManager(connection)
    model = OpenAIRealtime(api_key="test")
    model._client = cast(Any, _MockClient(manager))

    scope = model.session_prepare(
        instructions="",
        tools=ModelTools.none,
        context=(),
        output="text",
        config=_realtime_config(),
    )

    session = await scope.__aenter__()
    await session._writing(TextContent.of("hello"))

    assert connection.response.create_calls == 0
    assert connection.conversation.item.created_items == []

    await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_session_write_hidden_artifact_is_skipped() -> None:
    connection = _MockConnection()
    connection.session = _MockSessionResource()
    manager = _MockConnectionManager(connection)
    model = OpenAIRealtime(api_key="test")
    model._client = cast(Any, _MockClient(manager))

    scope = model.session_prepare(
        instructions="",
        tools=ModelTools.none,
        context=(),
        output="text",
        config=_realtime_config(),
    )

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

    assert set(current_items.keys()) == {
        "00000000-0000-0000-0000-000000000011",
        "00000000-0000-0000-0000-000000000012",
        "00000000-0000-0000-0000-000000000013",
        "00000000-0000-0000-0000-000000000014",
    }

    await _reset_context(
        (),
        current_items=current_items,
        connection=cast(Any, connection),
    )

    assert set(connection.conversation.item.deleted_ids) == {
        "00000000-0000-0000-0000-000000000011",
        "00000000-0000-0000-0000-000000000012",
        "00000000-0000-0000-0000-000000000013",
        "00000000-0000-0000-0000-000000000014",
    }
    assert current_items == {}
