from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from haiway import Pagination
from surrealdb.errors import parse_query_error

import draive.surreal.conversation_memory as surreal_memory
from draive.conversation.types import ConversationUserTurn
from draive.multimodal import MultimodalContent
from draive.surreal import SurrealConversationMemory, SurrealException, SurrealObject


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

    memory = SurrealConversationMemory.prepare(thread="thread-1")

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


@pytest.mark.asyncio
async def test_surreal_conversation_memory_migration_defines_table_and_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        _ = variables
        statements.append(statement)
        return ()

    monkeypatch.setattr(surreal_memory.Surreal, "execute", fake_execute)

    await SurrealConversationMemory.migrate()

    assert statements == [
        "DEFINE TABLE IF NOT EXISTS conversation_memory SCHEMALESS TYPE NORMAL;",
        "DEFINE INDEX IF NOT EXISTS conversation_memory_thread_idx "
        "ON TABLE conversation_memory FIELDS thread_id, created;",
        "DEFINE INDEX IF NOT EXISTS conversation_memory_cursor_idx "
        "ON TABLE conversation_memory FIELDS thread_id, identifier;",
    ]


@pytest.mark.asyncio
async def test_surreal_conversation_memory_propagates_missing_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table which was never defined nor written to is an error, its structures
    have to be defined with `migrate` upfront.
    """

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        _ = (statement, variables)
        raise SurrealException(
            "Surreal execution error: The table 'conversation_memory' does not exist"
        ) from parse_query_error(
            {
                "status": "ERR",
                "result": "The table 'conversation_memory' does not exist",
                "kind": "NotFound",
                "details": {"kind": "Table", "details": {"name": "conversation_memory"}},
            }
        )

    monkeypatch.setattr(surreal_memory.Surreal, "execute", fake_execute)

    memory = SurrealConversationMemory.prepare(thread=uuid4())
    with pytest.raises(SurrealException):
        await memory.recall()

    with pytest.raises(SurrealException):
        await memory.fetch(Pagination.of(limit=4))
