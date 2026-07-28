import json
from dataclasses import dataclass
from typing import cast
from uuid import UUID, uuid4

import pytest

import draive.postgres.agent_memory as postgres_agent_memory
from draive.agents import AgentMemory, AgentThread
from draive.models import ModelContext, ModelInput, ModelOutput
from draive.multimodal import MultimodalContent
from draive.postgres.agent_memory import PostgresAgentMemory


@dataclass(frozen=True)
class _FakeSnapshotRow:
    context: str

    def get_str(
        self,
        key: str,
        *,
        required: bool = False,
    ) -> str | None:
        if key == "context":
            return self.context

        if required:
            raise ValueError(f"Missing required value for '{key}'")

        return None


def _snapshot_row(context: ModelContext) -> _FakeSnapshotRow:
    return _FakeSnapshotRow(context=f"[{','.join(element.to_json() for element in context)}]")


def _thread(agent_uri: str = "agent://assistant") -> AgentThread:
    return AgentThread.of(uuid4(), agent_uri=agent_uri)


@pytest.mark.asyncio
async def test_postgres_agent_memory_recall_returns_latest_snapshot_with_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _thread()
    stored_context = (
        ModelInput.of(MultimodalContent.of("first")),
        ModelOutput.of(MultimodalContent.of("second")),
    )

    async def fake_fetch_one(
        statement: str,
        /,
        *args: object,
    ) -> _FakeSnapshotRow:
        assert "ORDER BY" in statement
        assert "LIMIT 1" in statement
        assert args == (thread.agent_uri, thread.identifier)
        return _snapshot_row(stored_context)

    monkeypatch.setattr(postgres_agent_memory.Postgres, "fetch_one", fake_fetch_one)

    memory: AgentMemory = PostgresAgentMemory.instance()
    incoming = ModelInput.of(MultimodalContent.of("third"))

    recalled = await memory.recall(thread=thread, context=(incoming,))

    assert [element.content.to_str() for element in recalled] == ["first", "second", "third"]
    assert isinstance(recalled[0], ModelInput)
    assert isinstance(recalled[1], ModelOutput)
    # snapshot round-trip preserves elements exactly
    assert recalled[:2] == stored_context
    assert recalled[2] is incoming


@pytest.mark.asyncio
async def test_postgres_agent_memory_recall_of_empty_thread_returns_input_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch_one(
        statement: str,
        /,
        *args: object,
    ) -> None:
        _ = (statement, args)

    monkeypatch.setattr(postgres_agent_memory.Postgres, "fetch_one", fake_fetch_one)

    memory: AgentMemory = PostgresAgentMemory.instance()
    incoming = ModelInput.of(MultimodalContent.of("hello"))

    assert await memory.recall(thread=_thread(), context=(incoming,)) == (incoming,)


@pytest.mark.asyncio
async def test_postgres_agent_memory_remember_inserts_snapshot_write_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _thread()
    fetched: list[tuple[str, tuple[object, ...]]] = []
    executed: list[tuple[str, tuple[object, ...]]] = []

    async def fake_fetch_one(
        statement: str,
        /,
        *args: object,
    ) -> None:
        fetched.append((statement, args))

    async def fake_execute(
        statement: str,
        /,
        *args: object,
    ) -> None:
        executed.append((statement, args))

    monkeypatch.setattr(postgres_agent_memory.Postgres, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(postgres_agent_memory.Postgres, "execute", fake_execute)

    memory: AgentMemory = PostgresAgentMemory.instance()
    context = (
        ModelInput.of(MultimodalContent.of("hello")),
        ModelOutput.of(MultimodalContent.of("world")),
    )

    await memory.remember(thread=thread, context=context)

    # write-only: no reads, no deletes, no locks - one snapshot insert
    assert fetched == []
    assert len(executed) == 1
    insert_statement, insert_args = executed[0]
    assert "INSERT INTO" in insert_statement
    assert "DELETE" not in insert_statement
    assert "pg_advisory" not in insert_statement

    assert insert_args[0] == thread.agent_uri
    assert insert_args[1] == thread.identifier
    assert isinstance(insert_args[2], UUID)

    # the snapshot serializes the full context exactly as provided
    snapshot = json.loads(cast(str, insert_args[3]))
    assert [value["kind"] for value in snapshot] == ["input", "output"]
    assert [postgres_agent_memory._element_from_value(value) for value in snapshot] == list(context)


@pytest.mark.asyncio
async def test_postgres_agent_memory_remember_accepts_disjoint_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _thread()
    snapshots: list[str] = []

    async def fake_fetch_one(
        statement: str,
        /,
        *args: object,
    ) -> _FakeSnapshotRow | None:
        _ = (statement, args)
        return _FakeSnapshotRow(context=snapshots[-1]) if snapshots else None

    async def fake_execute(
        statement: str,
        /,
        *args: object,
    ) -> None:
        _ = statement
        snapshots.append(cast(str, args[3]))

    monkeypatch.setattr(postgres_agent_memory.Postgres, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(postgres_agent_memory.Postgres, "execute", fake_execute)

    memory: AgentMemory = PostgresAgentMemory.instance()
    await memory.remember(
        thread=thread,
        context=(
            ModelInput.of(MultimodalContent.of("original")),
            ModelOutput.of(MultimodalContent.of("history")),
        ),
    )

    # e.g. a compaction step replaced the whole context with a summary
    summary = (ModelInput.of(MultimodalContent.of("summary of prior conversation")),)
    await memory.remember(thread=thread, context=summary)

    # both snapshots retained; the latest one is recalled whole, as provided
    assert len(snapshots) == 2
    incoming = ModelInput.of(MultimodalContent.of("next"))
    recalled = await memory.recall(thread=thread, context=(incoming,))
    assert recalled == (*summary, incoming)


def test_postgres_agent_memory_snapshot_value_round_trip() -> None:
    elements = (
        ModelInput.of(MultimodalContent.of("in")),
        ModelOutput.of(MultimodalContent.of("out")),
    )
    for element in elements:
        value = json.loads(element.to_json())
        assert postgres_agent_memory._element_from_value(value) == element

    with pytest.raises(ValueError):
        postgres_agent_memory._element_from_value({"kind": "other"})
