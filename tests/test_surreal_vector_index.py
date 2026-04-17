from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from haiway import AttributeRequirement, State, ctx

import draive.surreal.vector as surreal_vector
from draive.surreal.types import SurrealException, SurrealObject
from draive.surreal.vector import SurrealVectorIndex


class _Meta(State):
    kind: str


class _Doc(State):
    text: str
    group: str
    meta: _Meta


@pytest.mark.asyncio
async def test_surreal_vector_index_delete_scopes_requirements_to_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Mapping[str, Any]]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        calls.append((statement, variables))
        return ()

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    index = SurrealVectorIndex()
    async with ctx.scope("test.surreal.vector.delete"):
        await index.delete(
            _Doc,
            requirements=AttributeRequirement.equal("b", _Doc._.group),
        )

    assert calls == [("DELETE _Doc WHERE content.group = $_f0;", {"_f0": "b"})]


@pytest.mark.asyncio
async def test_surreal_vector_index_search_without_query_scopes_nested_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Mapping[str, Any]]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        calls.append((statement, variables))
        return (
            cast(
                SurrealObject,
                {
                    "content": {"text": "alpha", "group": "b", "meta": {"kind": "note"}},
                    "created": "2025-01-01T00:00:00Z",
                    "id": "doc:1",
                },
            ),
        )

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    index = SurrealVectorIndex()
    async with ctx.scope("test.surreal.vector.search.none"):
        results = await index.search(
            _Doc,
            query=None,
            requirements=AttributeRequirement.equal("note", _Doc._.meta.kind),
            limit=2,
        )

    assert len(results) == 1
    assert results[0].text == "alpha"
    assert len(calls) == 1
    statement, variables = calls[0]
    assert "WHERE content.meta.kind = $_f0" in statement
    assert variables["_f0"] == "note"
    assert variables["limit"] == 2


@pytest.mark.asyncio
async def test_surreal_vector_index_search_with_query_scopes_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Mapping[str, Any]]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        calls.append((statement, variables))
        return (
            cast(
                SurrealObject,
                {
                    "content": {"text": "alpha", "group": "a", "meta": {"kind": "note"}},
                    "embedding": [1.0, 0.0],
                    "distance": 0.0,
                },
            ),
        )

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    index = SurrealVectorIndex(search_effort=12)
    async with ctx.scope("test.surreal.vector.search.query"):
        results = await index.search(
            _Doc,
            query=[1.0, 0.0],
            requirements=AttributeRequirement.equal("a", _Doc._.group),
            limit=3,
        )

    assert len(results) == 1
    assert results[0].group == "a"
    assert len(calls) == 1
    statement, variables = calls[0]
    assert "WHERE (content.group = $_f0) AND embedding <|3,12|> $query" in statement
    assert variables["_f0"] == "a"
    assert variables["limit"] == 3
    assert variables["query"] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_surreal_vector_index_delete_ignores_missing_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        _ = (statement, variables)
        raise SurrealException("Surreal execution error: The table '_Doc' does not exist")

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    index = SurrealVectorIndex()
    async with ctx.scope("test.surreal.vector.delete.missing"):
        await index.delete(_Doc)
