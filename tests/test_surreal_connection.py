import asyncio
from typing import Any

import pytest
from surrealdb import AsyncEmbeddedSurrealConnection
from surrealdb.request_message.message import RequestMessage

from draive.surreal.connection import SurrealConnection


class _FakeEmbeddedConnection(AsyncEmbeddedSurrealConnection):
    """Embedded-connection stand-in tracking request concurrency.

    Overrides ``__init__`` (no engine spawned) and ``_send`` (no Rust extension
    involved) while keeping the real class so the fake matches the real connection type.
    """

    def __init__(self) -> None:  # deliberately no super().__init__()
        self.active_sends: int = 0
        self.max_concurrent_sends: int = 0

    async def _send(
        self,
        message: RequestMessage,
        process: str,
        bypass: bool = False,
    ) -> dict[str, Any]:
        self.active_sends += 1
        self.max_concurrent_sends = max(self.max_concurrent_sends, self.active_sends)
        # Suspend so any concurrently gathered request would be observed as
        # overlapping if serialization were missing.
        await asyncio.sleep(0.001)
        self.active_sends -= 1
        return {"result": []}


@pytest.mark.asyncio
async def test_embedded_connection_serializes_concurrent_statements() -> None:
    """Regression test for a live-fire finding: the embedded engine panics its Rust
    runtime (SIGABRT, killing the whole process) under concurrent requests - e.g.
    gathered writes into the same HNSW-indexed table, or a KNN search racing such a
    write. Every embedded ``_send`` must therefore be serialized through the
    connection's lock, no matter how many statements callers gather at once.
    """
    fake = _FakeEmbeddedConnection()
    connection = SurrealConnection(url="mem://")
    connection._connection = fake  # pyright: ignore[reportPrivateUsage]

    async with connection.prepare_session() as session:
        await asyncio.gather(*(session.execute("SELECT 1;") for _ in range(8)))

    assert fake.max_concurrent_sends == 1


def test_default_url_uses_embedded_engine_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("draive.surreal.connection.find_spec", lambda _name: object())  # pyright: ignore[reportUnknownLambdaType]

    connection = SurrealConnection()

    assert connection._url == "mem://"  # pyright: ignore[reportPrivateUsage]
    assert connection._embedded  # pyright: ignore[reportPrivateUsage]


def test_default_url_requires_explicit_url_without_embedded_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("draive.surreal.connection.find_spec", lambda _name: None)  # pyright: ignore[reportUnknownLambdaType]

    with pytest.raises(ValueError, match="surrealdb_embedded"):
        SurrealConnection()

    remote = SurrealConnection(url="ws://localhost:8000/rpc")
    assert not remote._embedded  # pyright: ignore[reportPrivateUsage]
