from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, cast

import pytest
from haiway import Alias, AttributeRequirement, State, ctx
from surrealdb.errors import parse_query_error

import draive.surreal.vector as surreal_vector
from draive.embedding import Embedded, TextEmbedding, VectorIndex
from draive.surreal import SurrealClient
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

    index = SurrealVectorIndex.prepare()
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

    index = SurrealVectorIndex.prepare()
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

    index = SurrealVectorIndex.prepare(search_effort=12)
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
async def test_surreal_vector_index_propagates_missing_table(
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
            "Surreal execution error: The table '_Doc' does not exist"
        ) from parse_query_error(
            {
                "status": "ERR",
                "result": "The table '_Doc' does not exist",
                "kind": "NotFound",
                "details": {"kind": "Table", "details": {"name": "_Doc"}},
            }
        )

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    index = SurrealVectorIndex.prepare()
    async with ctx.scope("test.surreal.vector.delete.missing"):
        with pytest.raises(SurrealException):
            await index.delete(_Doc)

        with pytest.raises(SurrealException):
            await index.search(_Doc, limit=4)


@pytest.mark.asyncio
async def test_surreal_vector_index_runs_against_embedded_engine() -> None:
    """Live regression test: `contained_in` operands are swapped, scoping its `lhs`
    used to turn the collection of values into a path string, and `text_match` used
    to emit an invalid function path - both failed only at the database.
    """

    async def embedding(
        values: Sequence[Any],
        /,
        attribute: Callable[[Any], Any] | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Any]]:
        _ = attribute, extra
        return [
            Embedded(
                value=value,
                vector=[float(len(str(value))), 1.0],
            )
            for value in values
        ]

    async with SurrealClient(url="mem://") as surreal:
        async with ctx.scope(
            "test.surreal.vector.embedded",
            surreal,
            TextEmbedding(embedding=embedding),
            SurrealVectorIndex.prepare(),
        ):
            await SurrealVectorIndex.migrate(_Doc, dimensions=2)
            await VectorIndex.index(
                _Doc,
                attribute=_Doc._.text,
                values=[
                    _Doc(text="alpha", group="a", meta=_Meta(kind="note")),
                    _Doc(text="beta", group="b", meta=_Meta(kind="task")),
                ],
            )

            assert {document.text for document in await VectorIndex.search(_Doc, limit=8)} == {
                "alpha",
                "beta",
            }

            assert tuple(
                document.text
                for document in await VectorIndex.search(
                    _Doc,
                    requirements=AttributeRequirement[_Doc].contained_in(("a",), _Doc._.group),
                    limit=8,
                )
            ) == ("alpha",)

            assert tuple(
                document.text
                for document in await VectorIndex.search(
                    _Doc,
                    requirements=AttributeRequirement[_Doc].text_match("beta", _Doc._.text),
                    limit=8,
                )
            ) == ("beta",)

            assert {
                document.text
                for document in await VectorIndex.search(
                    _Doc,
                    query="alpha",
                    requirements=AttributeRequirement[_Doc].contained_in(
                        ("a", "b"),
                        _Doc._.group,
                    ),
                    limit=8,
                )
            } == {"alpha", "beta"}

            await VectorIndex.delete(
                _Doc,
                requirements=AttributeRequirement[_Doc].contained_in(("a",), _Doc._.group),
            )
            assert tuple(document.text for document in await VectorIndex.search(_Doc, limit=8)) == (
                "beta",
            )


@pytest.mark.asyncio
async def test_surreal_vector_index_migration_defines_table_and_hnsw_index(
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

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    async with ctx.scope("test.surreal.vector.migrate"):
        await SurrealVectorIndex.migrate(
            _Doc,
            dimensions=1536,
            efc=150,
            m=12,
        )

    assert statements == [
        "DEFINE TABLE IF NOT EXISTS _Doc SCHEMALESS TYPE NORMAL;",
        "DEFINE INDEX IF NOT EXISTS _Doc_embedding_index "
        "ON TABLE _Doc FIELDS embedding "
        "HNSW DIMENSION 1536 TYPE F64 DIST COSINE EFC 150 M 12;",
    ]


@pytest.mark.asyncio
async def test_surreal_vector_index_migration_retries_definition_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a live-fire finding: defining a not yet existing index
    concurrently aborts with a key-value store conflict, therefore its definition
    is retried.
    """
    definitions: int = 0
    pending_conflicts: int = 1

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        nonlocal definitions, pending_conflicts
        _ = variables
        if not statement.startswith("DEFINE INDEX"):
            return ()

        definitions += 1
        if pending_conflicts > 0:
            pending_conflicts -= 1
            raise SurrealException(
                "Surreal execution error: There was a problem with the key-value store: "
                "Transaction conflict: Write conflict, retry the transaction."
            )

        return ()

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    async with ctx.scope("test.surreal.vector.migrate.conflict"):
        await SurrealVectorIndex.migrate(_Doc, dimensions=2)

    # the conflicting attempt and its retry
    assert definitions == 2
    assert pending_conflicts == 0


@pytest.mark.parametrize(
    "arguments",
    (
        {"dimensions": 0},
        {"dimensions": 2, "efc": 0},
        {"dimensions": 2, "m": 0},
        {"dimensions": 2, "vector_type": "bad type"},
        {"dimensions": 2, "distance": "bad distance"},
    ),
)
@pytest.mark.asyncio
async def test_surreal_vector_index_migration_rejects_invalid_arguments(
    arguments: Mapping[str, Any],
) -> None:
    async with ctx.scope("test.surreal.vector.migrate.invalid"):
        with pytest.raises(ValueError):
            await SurrealVectorIndex.migrate(_Doc, **arguments)


class _AliasedNested(State):
    inner_value: Annotated[str, Alias("inner")]


class _AliasedDoc(State):
    code: Annotated[str, Alias("reference")]
    text: str
    nested: _AliasedNested


@pytest.mark.asyncio
async def test_surreal_vector_index_scopes_aliased_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: documents are stored serialized, requirements written with
    python attribute names of aliased fields silently matched nothing.
    """
    calls: list[tuple[str, Mapping[str, Any]]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        calls.append((statement, variables))
        return ()

    monkeypatch.setattr(surreal_vector.Surreal, "execute", fake_execute)

    index = SurrealVectorIndex.prepare()
    async with ctx.scope("test.surreal.vector.aliased"):
        await index.delete(
            _AliasedDoc,
            requirements=AttributeRequirement.equal("doc-1", _AliasedDoc._.code),
        )
        await index.delete(
            _AliasedDoc,
            requirements=AttributeRequirement.equal("value", _AliasedDoc._.nested.inner_value),
        )

    assert calls == [
        ("DELETE _AliasedDoc WHERE content.reference = $_f0;", {"_f0": "doc-1"}),
        ("DELETE _AliasedDoc WHERE content.nested.inner = $_f0;", {"_f0": "value"}),
    ]


def test_surreal_filters_use_serialized_field_names() -> None:
    from draive.surreal.filters import prepare_filter

    clause, values = prepare_filter(
        AttributeRequirement[_AliasedDoc].equal("doc-1", _AliasedDoc._.code)
    )

    assert clause == "reference = $_f0"
    assert values == {"_f0": "doc-1"}
