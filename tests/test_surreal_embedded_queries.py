from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from haiway import AttributeRequirement, Paginated, Pagination, State, ctx

from draive.surreal import SurrealClient
from draive.surreal.filters import prepare_filter
from draive.surreal.state import Surreal, SurrealSession
from draive.surreal.templates import _fetch_template_rows  # pyright: ignore[reportPrivateUsage]
from draive.surreal.types import SurrealObject, SurrealValue
from draive.surreal.vector import (
    _content_scoped_requirements,  # pyright: ignore[reportPrivateUsage]
)

"""Regression tests running against the bundled embedded SurrealDB engine.

Every query shape covered here used to fail at the database - either with a parse
error or by silently doing nothing - while remaining invisible to the tests using
a fake statement executor.
"""


class _Item(State):
    name: str
    amount: int


class _Author(State):
    handle: str


class _Note(State):
    title: str


@asynccontextmanager
async def _embedded_session() -> AsyncGenerator[SurrealSession]:
    async with SurrealClient(url="mem://") as surreal:
        async with ctx.scope("surreal-embedded-test", surreal):
            async with Surreal.prepare_session() as session:
                yield session


@pytest.mark.asyncio
async def test_fetch_reads_models_without_declared_identifier() -> None:
    """Regression test: `SELECT *` returns the implicit `id` which no model
    declares, while declaring it would in turn break `create`.
    """
    async with _embedded_session() as session:
        await session.define_table(_Item)
        await session.create(_Item(name="alpha", amount=1))
        await session.create(_Item(name="beta", amount=2))

        page: Paginated[_Item] = await session.fetch(_Item)
        assert {item.name for item in page.items} == {"alpha", "beta"}

        ordered: Paginated[_Item] = await session.fetch(
            _Item,
            order=_Item._.amount,
        )
        assert tuple(item.name for item in ordered.items) == ("beta", "alpha")

        first: Paginated[_Item] = await session.fetch(
            _Item,
            pagination=Pagination.of(limit=1),
        )
        assert len(first.items) == 1
        assert first.pagination.token == 1


@pytest.mark.asyncio
async def test_upsert_reads_back_the_stored_model() -> None:
    async with _embedded_session() as session:
        await session.define_table(_Item)
        await session.upsert(_Item(name="alpha", amount=1), identifier="alpha")
        await session.upsert(_Item(name="alpha", amount=2), identifier="alpha")

        page: Paginated[_Item] = await session.fetch(_Item)
        assert page.items == (_Item(name="alpha", amount=2),)


@pytest.mark.asyncio
async def test_text_match_requirement_filters_records() -> None:
    """Regression test: `string(...)` is not a valid SurrealQL function path,
    every `text_match` requirement used to end with a parse error.
    """
    async with _embedded_session() as session:
        await session.define_table(_Item)
        await session.create(_Item(name="alpha beta", amount=1))
        await session.create(_Item(name="gamma", amount=2))

        page: Paginated[_Item] = await session.fetch(
            _Item,
            requirements=AttributeRequirement[_Item].text_match("alpha", _Item._.name),
        )
        assert tuple(item.name for item in page.items) == ("alpha beta",)


@pytest.mark.asyncio
async def test_contained_in_requirement_filters_records() -> None:
    async with _embedded_session() as session:
        await session.define_table(_Item)
        await session.create(_Item(name="alpha", amount=1))
        await session.create(_Item(name="beta", amount=2))

        page: Paginated[_Item] = await session.fetch(
            _Item,
            requirements=AttributeRequirement[_Item].contained_in((2, 5), _Item._.amount),
        )
        assert tuple(item.name for item in page.items) == ("beta",)


@pytest.mark.asyncio
async def test_delete_removes_records_by_table() -> None:
    """Regression test: SurrealDB refuses to run DELETE using a plain string
    variable, both delete paths used to leave every record in place.
    """
    async with _embedded_session() as session:
        await session.define_table(_Item)
        await session.create(_Item(name="alpha", amount=1))
        await session.create(_Item(name="beta", amount=2))

        await session.delete(
            _Item,
            requirements=AttributeRequirement[_Item].equal(2, _Item._.amount),
        )
        assert tuple(item.name for item in (await session.fetch(_Item)).items) == ("alpha",)

        await session.delete(_Item)
        assert (await session.fetch(_Item)).items == ()


@pytest.mark.asyncio
async def test_related_lists_targets_in_both_directions() -> None:
    """Regression test: SurrealDB requires each ORDER BY idiom to appear within
    the projection, the query used to end with a parse error.
    """
    async with _embedded_session() as session:
        await session.define_table(_Author)
        await session.define_table(_Note)
        await session.define_relation(
            "wrote",
            from_table=_Author,
            to_table=_Note,
        )
        await session.create(_Author(handle="author"), identifier="author")
        await session.create(_Note(title="first"), identifier="first")
        await session.create(_Note(title="second"), identifier="second")
        await session.relate(_Author, "author", "wrote", _Note, "first")
        await session.relate(_Author, "author", "wrote", _Note, "second")

        outgoing: Paginated[_Note] = await session.related(
            _Author,
            "author",
            "wrote",
            _Note,
        )
        assert {note.title for note in outgoing.items} == {"first", "second"}

        incoming: Paginated[_Author] = await session.related(
            _Note,
            "first",
            "wrote",
            _Author,
            direction="in",
        )
        assert incoming.items == (_Author(handle="author"),)

        paginated: Paginated[_Note] = await session.related(
            _Author,
            "author",
            "wrote",
            _Note,
            pagination=Pagination.of(limit=1),
        )
        assert len(paginated.items) == 1
        assert paginated.pagination.token == 1


@pytest.mark.asyncio
async def test_templates_listing_returns_the_latest_revision_of_each_identifier() -> None:
    """Regression test: the inner `SELECT VALUE id ... ORDER BY` used to parse-error
    while `SELECT VALUE identifier ... GROUP BY identifier` yielded a single NONE
    instead of the distinct identifiers.
    """
    async with _embedded_session() as session:
        revisions: Sequence[tuple[str, int, str]] = (
            ("alpha", 2020, "alpha-old"),
            ("alpha", 2021, "alpha-new"),
            ("beta", 2020, "beta-only"),
            ("gamma", 2019, "gamma-old"),
            ("gamma", 2022, "gamma-new"),
        )
        for identifier, year, content in revisions:
            await session.execute(
                """
                CREATE templates CONTENT {
                    identifier: $identifier,
                    updated: $updated,
                    description: $description,
                    content: $content,
                    variables: {},
                    meta: {}
                };
                """,
                identifier=identifier,
                updated=datetime(year, 1, 1, tzinfo=UTC),
                description=content,
                content=content,
            )

        rows: Sequence[SurrealObject] = await _fetch_template_rows(
            after_identifier=None,
            limit=10,
        )
        assert [(row["identifier"], row["description"]) for row in rows] == [
            ("alpha", "alpha-new"),
            ("beta", "beta-only"),
            ("gamma", "gamma-new"),
        ]

        rows = await _fetch_template_rows(
            after_identifier="alpha",
            limit=10,
        )
        assert [row["identifier"] for row in rows] == ["beta", "gamma"]

        rows = await _fetch_template_rows(
            after_identifier=None,
            limit=2,
        )
        assert [row["identifier"] for row in rows] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_vector_index_requirements_are_scoped_to_the_content_field() -> None:
    """Regression test: 'contained_in' operands are swapped, scoping its `lhs`
    turned the collection of values into a path string and made the whole
    requirement unusable.
    """
    scoped: AttributeRequirement[_Item] | None = _content_scoped_requirements(
        AttributeRequirement[_Item].contained_in((1, 2), _Item._.amount)
    )
    assert scoped is not None
    assert scoped.lhs == (1, 2)
    assert str(scoped.rhs) == "content.amount"

    filter_clause: str
    filter_variables: Mapping[str, SurrealValue]
    filter_clause, filter_variables = prepare_filter(scoped)
    assert filter_clause == "content.amount INSIDE $_f0"

    async with _embedded_session() as session:
        for name, amount in (("alpha", 1), ("beta", 5)):
            await session.execute(
                "CREATE _Item SET content = $content;",
                content=cast(Any, _Item(name=name, amount=amount).to_mapping()),
            )

        rows: Sequence[SurrealObject] = await session.execute(
            f"SELECT content FROM _Item WHERE {filter_clause};",  # nosec: B608
            **cast(Any, filter_variables),
        )
        assert [cast(Mapping[str, Any], row["content"])["name"] for row in rows] == ["alpha"]


@pytest.mark.asyncio
async def test_records_holding_a_duration_are_readable() -> None:
    """Regression test: a decoded SurrealDB `Duration` used to raise, making any
    record holding a duration field unreadable.
    """
    async with _embedded_session() as session:
        await session.execute("CREATE spans:1 SET elapsed = 1s;")

        rows: Sequence[SurrealObject] = await session.execute("SELECT * OMIT id FROM spans;")
        assert rows == ({"elapsed": timedelta(seconds=1)},)
