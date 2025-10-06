import importlib.util
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from haiway import AttributeRequirement
from haiway.postgres import Postgres

from draive.embedding import TextEmbedding
from draive.embedding.types import Embedded
from draive.parameters import DataModel
from draive.utils import VectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

_VECTOR_INDEX_SPEC = importlib.util.spec_from_file_location(
    "_postgres_vector_index",
    SRC_PATH / "draive" / "postgres" / "vector_index.py",
)
_VECTOR_INDEX_MODULE = importlib.util.module_from_spec(_VECTOR_INDEX_SPEC)
assert _VECTOR_INDEX_SPEC and _VECTOR_INDEX_SPEC.loader
_VECTOR_INDEX_SPEC.loader.exec_module(_VECTOR_INDEX_MODULE)
PostgresVectorIndex = _VECTOR_INDEX_MODULE.PostgresVectorIndex


class _FakeTransaction:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection

    async def __aenter__(self) -> "_FakeConnection":
        return self._connection

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, statement: str, *params: Any) -> None:
        self.executed.append((statement, params))

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

    async def __aenter__(self) -> "_FakeConnection":
        return self

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        return None


class _FakeAcquire:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        return None


class Chunk(DataModel):
    text: str


def _vector_index() -> VectorIndex:
    return PostgresVectorIndex()


def _install_text_embedding_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vector: Iterable[float],
) -> None:
    vector_tuple: tuple[float, ...] = tuple(vector)

    async def _stub(content: Any, /, **_: Any) -> Any:
        if isinstance(content, Iterable) and not isinstance(content, (str, bytes, bytearray)):
            return [Embedded(value=value, vector=vector_tuple) for value in content]
        return Embedded(value=content, vector=vector_tuple)

    monkeypatch.setattr(TextEmbedding, "embed", _stub)


@pytest.mark.asyncio
async def test_index_persists_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection()
    monkeypatch.setattr(Postgres, "acquire_connection", lambda: _FakeAcquire(connection))
    _install_text_embedding_stub(monkeypatch, vector=(1.0, 0.0, 0.0))

    index = _vector_index()

    await index.index(
        Chunk,
        attribute=lambda chunk: chunk.text,
        values=[
            Chunk(text="first"),
            Chunk(text="second"),
        ],
        namespace="docs",
    )

    assert len(connection.executed) == 2
    for statement, params in connection.executed:
        compact_statement = " ".join(statement.split())
        assert "INSERT INTO chunk" in compact_statement
        assert "embedding" in compact_statement
        assert params[0] == (1.0, 0.0, 0.0)
        payload = Chunk.from_json(params[1])
        assert payload.text in {"first", "second"}
        assert isinstance(params[3], datetime)


@pytest.mark.asyncio
async def test_search_orders_by_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_fetch(statement: str, *params: Any) -> tuple[dict[str, Any], ...]:
        captured["statement"] = statement
        captured["params"] = params
        return (
            {
                "payload": Chunk(text="relevant").to_json(),
                "embedding": (1.0, 0.0, 0.0),
            },
        )

    monkeypatch.setattr(Postgres, "fetch", _fake_fetch)
    _install_text_embedding_stub(monkeypatch, vector=(1.0, 0.0, 0.0))

    index = _vector_index()

    results = await index.search(
        Chunk,
        query="prompt",
        limit=1,
        score_threshold=0.5,
    )

    assert results == (Chunk(text="relevant"),)
    normalized = " ".join(captured["statement"].split())
    assert "WHERE" in normalized
    assert "embedding <#> $1" in normalized
    assert normalized.endswith("LIMIT $3;")
    assert captured["params"][0] == pytest.approx((1.0, 0.0, 0.0))
    assert captured["params"][1] == pytest.approx(0.5)
    assert captured["params"][2] == 1


@pytest.mark.asyncio
async def test_search_applies_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_fetch(statement: str, *params: Any) -> tuple[dict[str, Any], ...]:
        captured["statement"] = statement
        captured["params"] = params
        return (
            {
                "payload": Chunk(text="match").to_json(),
                "embedding": (1.0, 0.0, 0.0),
            },
        )

    monkeypatch.setattr(Postgres, "fetch", _fake_fetch)
    _install_text_embedding_stub(monkeypatch, vector=(1.0, 0.0, 0.0))

    index = _vector_index()
    requirement = AttributeRequirement.equal("match", Chunk._.text)

    results = await index.search(
        Chunk,
        query="prompt",
        requirements=requirement,
        limit=2,
    )

    assert results == (Chunk(text="match"),)
    normalized = " ".join(captured["statement"].split())
    assert normalized.count("WHERE") == 1
    assert "payload #>>" in normalized
    assert captured["params"][0] == pytest.approx((1.0, 0.0, 0.0))
    assert captured["params"][1] == "match"
    assert captured["params"][2] == 2


@pytest.mark.asyncio
async def test_delete_uses_resolved_where_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def _fake_execute(statement: str, *params: Any) -> None:
        execute_calls.append((statement, params))

    monkeypatch.setattr(Postgres, "execute", _fake_execute)

    index = _vector_index()
    requirement = AttributeRequirement.equal("match", Chunk._.text)

    await index.delete(
        Chunk,
        requirements=requirement,
    )

    assert len(execute_calls) == 1
    statement, params = execute_calls[0]
    normalized = " ".join(statement.split())
    assert normalized.startswith("DELETE FROM chunk WHERE")
    assert normalized.count("WHERE") == 1
    assert params == ("match",)
