from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

import pytest

import draive.postgres.agent_memory as postgres_agent_memory
from draive.agents import AgentIdentity, AgentMemory, AgentThread
from draive.models import ModelInput, ModelOutput
from draive.multimodal import MultimodalContent
from draive.postgres.agent_memory import PostgresAgentMemory


@dataclass(frozen=True)
class _FakeRow:
    kind: str
    payload: str

    def __getitem__(
        self,
        key: str,
    ) -> str:
        if key != "payload":
            raise KeyError(key)

        return self.payload

    def get_str(
        self,
        key: str,
        *,
        required: bool = False,
    ) -> str | None:
        if key == "kind":
            return self.kind

        if key == "payload":
            return self.payload

        if required:
            raise ValueError(f"Missing required value for '{key}'")

        return None


@pytest.mark.asyncio
async def test_postgres_agent_memory_recall_appends_incoming_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = AgentIdentity.of(name="assistant")
    thread = AgentThread.of(uuid4())
    stored_rows = (
        _FakeRow(kind="input", payload=ModelInput.of(MultimodalContent.of("first")).to_json()),
        _FakeRow(kind="output", payload=ModelOutput.of(MultimodalContent.of("second")).to_json()),
    )

    async def fake_fetch(
        statement: str,
        /,
        *args: object,
    ) -> Sequence[_FakeRow]:
        _ = statement
        assert args == (identity.uri, thread.identifier)
        return stored_rows

    monkeypatch.setattr(postgres_agent_memory.Postgres, "fetch", fake_fetch)

    memory: AgentMemory = PostgresAgentMemory.prepare(identity)
    incoming = ModelInput.of(MultimodalContent.of("third"))

    recalled = await memory.recall(thread=thread, input=incoming)

    assert len(recalled) == 3
    assert isinstance(recalled[0], ModelInput)
    assert recalled[0].content.to_str() == "first"
    assert isinstance(recalled[1], ModelOutput)
    assert recalled[1].content.to_str() == "second"
    assert recalled[2] is incoming


@pytest.mark.asyncio
async def test_postgres_agent_memory_remember_replaces_thread_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = AgentIdentity.of(name="assistant")
    thread = AgentThread.of(uuid4())
    executed: list[tuple[str, tuple[object, ...]]] = []

    class _FakeConnection:
        async def execute(
            self,
            statement: str,
            /,
            *args: object,
        ) -> None:
            executed.append((statement, args))

        def transaction(self) -> _FakeTransaction:
            return _FakeTransaction()

    class _FakeTransaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            return None

    class _FakeConnectionContext:
        async def __aenter__(self) -> _FakeConnection:
            return _FakeConnection()

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            return None

    def fake_acquire_connection(*_args: Any, **_kwargs: Any) -> _FakeConnectionContext:
        return _FakeConnectionContext()

    monkeypatch.setattr(
        postgres_agent_memory.Postgres,
        "acquire_connection",
        fake_acquire_connection,
    )

    memory: AgentMemory = PostgresAgentMemory.prepare(identity)
    context = (
        ModelInput.of(MultimodalContent.of("hello")),
        ModelOutput.of(MultimodalContent.of("world")),
    )

    await memory.remember(thread=thread, context=context)

    assert len(executed) == 3
    lock_statement, lock_args = executed[0]
    assert "pg_advisory_xact_lock" in lock_statement
    assert lock_args == (identity.uri, thread.identifier)

    delete_statement, delete_args = executed[1]
    assert "DELETE FROM" in delete_statement
    assert delete_args == (identity.uri, thread.identifier)

    insert_statement, insert_args = executed[2]
    assert "INSERT INTO" in insert_statement
    assert "UNNEST" in insert_statement
    assert insert_args[0] == identity.uri
    assert insert_args[1] == thread.identifier

    identifiers = insert_args[2]
    assert isinstance(identifiers, list)
    assert len(identifiers) == 2
    assert all(isinstance(identifier, UUID) for identifier in identifiers)
    assert identifiers[0] != identifiers[1]

    timestamps = insert_args[3]
    assert isinstance(timestamps, list)
    assert len(timestamps) == 2
    assert all(isinstance(timestamp, datetime) for timestamp in timestamps)
    # `created` must be fabricated as a strictly increasing sequence rather than reused as-is
    # (e.g. PostgreSQL's `CURRENT_TIMESTAMP` is fixed for the whole transaction) - `_recall`
    # relies on it to reconstruct insertion order. Regression guard for that ordering bug.
    assert timestamps[1] > timestamps[0]

    assert insert_args[4] == ["input", "output"]
    assert insert_args[5] == [element.to_json() for element in context]
