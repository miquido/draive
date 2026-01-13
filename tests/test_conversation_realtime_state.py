from typing import Any

import pytest
from haiway import ctx

from draive.conversation.realtime import RealtimeConversation
from draive.conversation.realtime.types import (
    RealtimeConversationSession,
    RealtimeConversationSessionScope,
)
from draive.conversation.state import ConversationMemory
from draive.models import ModelInstructions, ModelSessionOutputSelection
from draive.multimodal import TextContent
from draive.tools import Toolbox


@pytest.mark.asyncio
async def test_realtime_conversation_uses_injected_preparing() -> None:
    captured: dict[str, Any] = {}

    async def read() -> TextContent:
        return TextContent.of("ok")

    async def write(
        input: TextContent,  # noqa: A002
    ) -> None:
        _ = input

    async def open_session() -> RealtimeConversationSession:
        return RealtimeConversationSession(reading=read, writing=write)

    async def close_session(
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)

    expected_scope = RealtimeConversationSessionScope(
        opening=open_session,
        closing=close_session,
    )

    async def preparing(
        *,
        instructions: ModelInstructions,
        toolbox: Toolbox,
        memory: ConversationMemory,
        output: ModelSessionOutputSelection,
        **extra: Any,
    ) -> RealtimeConversationSessionScope:
        captured["instructions"] = instructions
        captured["toolbox"] = toolbox
        captured["memory"] = memory
        captured["output"] = output
        captured["extra"] = extra
        return expected_scope

    memory = ConversationMemory.disabled

    async with ctx.scope("test", RealtimeConversation(preparing=preparing)):
        scope = await RealtimeConversation.prepare(
            instructions="abc",
            tools=Toolbox.empty,
            memory=memory,
            output="text",
            marker="x",
        )

    assert scope is expected_scope
    assert captured["instructions"] == "abc"
    assert captured["memory"] is memory
    assert captured["output"] == "text"
    assert captured["extra"] == {"marker": "x"}
    assert isinstance(captured["toolbox"], Toolbox)
