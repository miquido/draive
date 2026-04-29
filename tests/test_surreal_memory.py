from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

import draive.surreal.memory as surreal_memory
from draive.conversation.types import ConversationUserTurn
from draive.multimodal import MultimodalContent
from draive.surreal.memory import SurrealConversationMemory
from draive.surreal.types import SurrealObject


@pytest.mark.asyncio
async def test_surreal_conversation_memory_remember_flattens_execute_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = ConversationUserTurn.of(
        MultimodalContent.of("content"),
        identifier=uuid4(),
        created=datetime(2026, 4, 24, tzinfo=UTC),
    )
    execution_variables: list[Mapping[str, Any]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        if statement.lstrip().startswith("DEFINE "):
            return ()
        assert "thread_id = $thread_id" in statement
        execution_variables.append(variables)
        return ()

    monkeypatch.setattr(surreal_memory.Surreal, "execute", fake_execute)

    memory = SurrealConversationMemory(thread="thread-1")

    await memory.remember(turn)

    assert execution_variables == [
        {
            "thread_id": "thread-1",
            "turn": "user",
            "identifier": str(turn.identifier),
            "payload": turn.to_json(),
            "created": turn.created,
        }
    ]
