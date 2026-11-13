from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import pytest
from haiway import Pagination, ctx
from haiway.postgres.types import PostgresTransactionContext

from draive.conversation.types import ConversationTurn, ConversationUserTurn
from draive.multimodal import MultimodalContent
from draive.postgres.conversation_memory import PostgresConversationMemory

from ._postgres_connection import postgres_states


@dataclass(frozen=True)
class _FakeRow:
    payload: str
    created: datetime
    identifier: UUID

    def __getitem__(
        self,
        key: str,
    ) -> str:
        if key != "payload":
            raise KeyError(key)

        return self.payload

    def get_datetime(
        self,
        key: str,
        *,
        required: bool = False,
    ) -> datetime | None:
        if key == "created":
            return self.created

        if required:
            raise ValueError(f"Missing required value for '{key}'")

        return None

    def get_uuid(
        self,
        key: str,
        *,
        required: bool = False,
    ) -> UUID | None:
        if key == "identifier":
            return self.identifier

        if required:
            raise ValueError(f"Missing required value for '{key}'")

        return None


def _row(
    turn: ConversationTurn,
) -> _FakeRow:
    return _FakeRow(
        payload=turn.to_json(),
        created=turn.created,
        identifier=turn.identifier,
    )


@pytest.mark.asyncio
async def test_postgres_conversation_memory_fetch_uses_pagination_token(
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
    rows = tuple(_row(turn) for turn in turns)

    async def fake_fetch(
        statement: str,
        /,
        *args: object,
    ) -> Sequence[_FakeRow]:
        _ = statement
        match args:
            case ("thread-1", 2):
                return rows[1:]

            case ("thread-1", identifier, 2):
                assert identifier == turns[1].identifier
                return rows[:1]

            case _:
                raise AssertionError(f"Unexpected fetch arguments: {args!r}")

    async with ctx.scope("test", *postgres_states(fetch=fake_fetch)):
        memory = PostgresConversationMemory.prepare(thread="thread-1")

        page_1 = await memory.fetch(Pagination.of(limit=2))
        assert [turn.content[0].to_str() for turn in page_1.items] == ["second", "third"]
        assert page_1.pagination.token == turns[1].identifier

        page_2 = await memory.fetch(page_1.pagination)
        assert [turn.content[0].to_str() for turn in page_2.items] == ["first"]
        assert page_2.pagination.token is None


@pytest.mark.asyncio
async def test_postgres_conversation_memory_recall_uses_latest_turns_in_order(
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
    rows = tuple(_row(turn) for turn in turns)

    async def fake_fetch(
        statement: str,
        /,
        *args: object,
    ) -> Sequence[_FakeRow]:
        _ = statement
        match args:
            case ("thread-1", 2):
                return rows[1:]

            case _:
                raise AssertionError(f"Unexpected fetch arguments: {args!r}")

    async with ctx.scope("test", *postgres_states(fetch=fake_fetch)):
        memory = PostgresConversationMemory.prepare(thread="thread-1")

        context = await memory.recall(Pagination.of(limit=2))

        assert [element.content.to_str() for element in context] == ["second", "third"]


@pytest.mark.asyncio
async def test_postgres_conversation_memory_remember_reuses_bound_connection() -> None:
    events: list[str] = []

    async def fake_execute(statement: str, /, *args: object) -> None:
        events.append("execute")

    @asynccontextmanager
    async def transaction() -> AsyncGenerator[None]:
        events.append("begin")
        yield
        events.append("commit")

    async with ctx.scope(
        "test",
        *postgres_states(
            execute=fake_execute,
            transaction_context=lambda: cast(PostgresTransactionContext, transaction()),
        ),
    ):
        memory = PostgresConversationMemory.prepare(thread="thread-1")
        await memory.remember(
            ConversationUserTurn.of(MultimodalContent.of("first")),
            ConversationUserTurn.of(MultimodalContent.of("second")),
        )

    assert events == ["begin", "execute", "execute", "commit"]
