from collections.abc import Sequence
from typing import Any

import pytest
from haiway import AttributeRequirement, State

import draive.postgres.vector_index as postgres_vector_index
from draive.embedding import VectorIndex
from draive.postgres.vector_index import PostgresVectorIndex


class _Document(State):
    name: str
    count: int


def _fake_fetch(
    statements: list[str],
    arguments: list[tuple[Any, ...]],
) -> Any:
    async def fetch(
        statement: str,
        /,
        *args: Any,
    ) -> Sequence[Any]:
        statements.append(statement)
        arguments.append(args)
        return ()

    return fetch


def _fake_execute(
    statements: list[str],
    arguments: list[tuple[Any, ...]],
) -> Any:
    async def execute(
        statement: str,
        /,
        *args: Any,
    ) -> None:
        statements.append(statement)
        arguments.append(args)

    return execute


@pytest.mark.asyncio
async def test_search_orders_by_cosine_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `<#>` is the negative inner product, which makes the
    `1.0 - score_threshold` predicate meaningless and can never be served by the
    recommended `vector_cosine_ops` index - `<=>` is the cosine distance.
    """
    statements: list[str] = []
    arguments: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        postgres_vector_index.Postgres,
        "fetch",
        _fake_fetch(statements, arguments),
    )
    index: VectorIndex = PostgresVectorIndex.prepare()

    await index.searching(
        _Document,
        query=[1.0, 0.0],
        score_threshold=0.8,
        limit=4,
    )

    assert len(statements) == 1
    assert "embedding <=> $1" in statements[0]
    assert "<#>" not in statements[0]
    assert "ORDER BY embedding <=> $1" in statements[0]
    # the threshold is converted to a cosine distance
    assert arguments[0][1] == pytest.approx(1.0 - 0.8)


@pytest.mark.asyncio
async def test_search_rejects_unsafe_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []
    arguments: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        postgres_vector_index.Postgres,
        "fetch",
        _fake_fetch(statements, arguments),
    )
    monkeypatch.setattr(
        postgres_vector_index.Postgres,
        "execute",
        _fake_execute(statements, arguments),
    )
    index: VectorIndex = PostgresVectorIndex.prepare()

    class _Unsafe(State):
        name: str

    _Unsafe.__name__ = "Document; DROP TABLE other"

    with pytest.raises(ValueError, match="Invalid Postgres identifier"):
        await index.searching(_Unsafe)

    with pytest.raises(ValueError, match="Invalid Postgres identifier"):
        await index.deleting(_Unsafe)

    with pytest.raises(ValueError, match="Invalid Postgres identifier"):
        await index.deleting(
            _Unsafe,
            requirements=AttributeRequirement[Any].equal("draive", _Unsafe._.name),
        )

    assert statements == []


@pytest.mark.asyncio
async def test_reranking_search_decodes_the_embedding_as_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: asyncpg has no codec for the pgvector `VECTOR` type, so a
    plain `SELECT embedding` hands the reranking its text representation instead of
    floats - the column has to be selected cast to `REAL[]`.
    """
    statements: list[str] = []
    arguments: list[tuple[Any, ...]] = []

    async def fetch(
        statement: str,
        /,
        *args: Any,
    ) -> Sequence[Any]:
        statements.append(statement)
        arguments.append(args)
        # what asyncpg returns for a `REAL[]` column
        return (
            {"embedding": [1.0, 0.0], "payload": '{"name": "first", "count": 1}'},
            {"embedding": [0.0, 1.0], "payload": '{"name": "second", "count": 2}'},
        )

    monkeypatch.setattr(postgres_vector_index.Postgres, "fetch", fetch)
    index: VectorIndex = PostgresVectorIndex.prepare()

    results: Sequence[_Document] = await index.searching(
        _Document,
        query=[1.0, 0.0],
        rerank=True,
        limit=2,
    )

    assert "embedding::REAL[] AS embedding" in statements[0]
    # the reranking consumed the decoded vectors and returned both documents
    assert {result.name for result in results} == {"first", "second"}
