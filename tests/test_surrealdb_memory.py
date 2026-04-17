from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import pytest
from haiway import Pagination

import draive.surreal.memory as surreal_memory
from draive.conversation.types import ConversationTurn, ConversationUserTurn
from draive.multimodal import MultimodalContent
from draive.surreal.memory import SurrealConversationMemory
from draive.surreal.types import SurrealObject


def _row(
    turn: ConversationTurn,
) -> SurrealObject:
    return cast(
        SurrealObject,
        {
            "payload": turn.to_json(),
            "created": turn.created,
            "identifier": str(turn.identifier),
        },
    )


@pytest.mark.asyncio
async def test_surrealdb_conversation_memory_fetch_uses_cursor_lookup_without_multi_statement_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_time = datetime(2026, 3, 13, tzinfo=UTC)
    turns: Sequence[ConversationTurn] = (
        ConversationUserTurn.of(
            MultimodalContent.of("first"),
            identifier=uuid4(),
            created=base_time,
        ),
        ConversationUserTurn.of(
            MultimodalContent.of("second"),
            identifier=uuid4(),
            created=base_time + timedelta(seconds=1),
        ),
        ConversationUserTurn.of(
            MultimodalContent.of("third"),
            identifier=uuid4(),
            created=base_time + timedelta(seconds=2),
        ),
    )
    rows: tuple[SurrealObject, ...] = tuple(_row(turn) for turn in turns)
    execution_calls: list[Mapping[str, Any]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        if statement.lstrip().startswith("DEFINE "):
            return ()
        execution_calls.append(variables)

        match variables:
            case {
                "thread_id": "thread-1",
                "cursor_created": created,
                "cursor_identifier": identifier,
                "limit": 3,
            }:
                assert created == turns[1].created
                assert identifier == str(turns[1].identifier)
                assert "LET $cursor" not in statement
                return rows[:1]

            case {"thread_id": "thread-1", "cursor": cursor}:
                assert cursor == str(turns[1].identifier)
                assert "LET $cursor" not in statement
                return (
                    cast(
                        SurrealObject,
                        {
                            "created": turns[1].created,
                            "identifier": str(turns[1].identifier),
                        },
                    ),
                )

            case {"thread_id": "thread-1", "limit": 3}:
                assert "LET $cursor" not in statement
                return rows[::-1]

            case _:
                raise AssertionError(f"Unexpected query variables: {variables!r}")

    monkeypatch.setattr(surreal_memory.Surreal, "execute", fake_execute)

    memory = SurrealConversationMemory(thread="thread-1")

    page_1 = await memory.fetch(Pagination.of(limit=2))
    assert [turn.content[0].to_str() for turn in page_1.items] == ["second", "third"]
    assert page_1.pagination.token == f"conversation_memory:cursor:{turns[1].identifier}"

    page_2 = await memory.fetch(page_1.pagination)
    assert [turn.content[0].to_str() for turn in page_2.items] == ["first"]
    assert page_2.pagination.token is None
    assert execution_calls == [
        {"thread_id": "thread-1", "limit": 3},
        {"thread_id": "thread-1", "cursor": str(turns[1].identifier)},
        {
            "thread_id": "thread-1",
            "cursor_created": turns[1].created,
            "cursor_identifier": str(turns[1].identifier),
            "limit": 3,
        },
    ]
