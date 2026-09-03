import warnings
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import pytest
from haiway import Alias, AttributeRequirement, Paginated, Pagination, State, ctx
from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchAny, Range

from draive.embedding import Embedded, VectorIndex
from draive.qdrant import Qdrant, QdrantClient, QdrantException
from draive.qdrant.filters import prepare_filter
from draive.qdrant.index import QdrantVectorIndex


def test_qdrant_vector_index_call_remains_compatible() -> None:
    with pytest.raises(
        RuntimeError,
        match="QdrantVectorIndex instantiation is forbidden",
    ):
        QdrantVectorIndex()


def test_qdrant_vector_index_prepare_returns_vector_index() -> None:
    assert isinstance(QdrantVectorIndex.prepare(), VectorIndex)


class _Document(State):
    text: str
    tag: str
    tags: Sequence[str]


@asynccontextmanager
async def _qdrant_scope() -> AsyncGenerator[None]:
    async with ctx.scope(
        "qdrant-test",
        QdrantVectorIndex.prepare(),
        disposables=[QdrantClient(in_memory=True)],
    ):
        await Qdrant.create_collection(
            _Document,
            vector_size=2,
            in_ram=True,
        )
        await Qdrant.store(
            _Document,
            objects=[
                Embedded(
                    value=_Document(
                        text=f"text-{index}",
                        tag="even" if index % 2 == 0 else "odd",
                        tags=(f"tag-{index}",),
                    ),
                    vector=[float(index), 1.0],
                )
                for index in range(4)
            ],
        )
        yield


@pytest.mark.asyncio
async def test_qdrant_fetch_continues_past_first_page() -> None:
    async with _qdrant_scope():
        first: Paginated[_Document] = await Qdrant.fetch(
            _Document,
            pagination=Pagination.of(limit=2),
        )
        assert len(first.items) == 2
        assert first.pagination.token is not None

        second: Paginated[_Document] = await Qdrant.fetch(
            _Document,
            pagination=first.pagination,
        )
        assert len(second.items) == 2
        assert {element.text for element in first.items}.isdisjoint(
            element.text for element in second.items
        )


@pytest.mark.asyncio
async def test_qdrant_vector_index_search_without_query_fetches_all() -> None:
    async with _qdrant_scope():
        results: Sequence[_Document] = await VectorIndex.search(
            _Document,
            limit=8,
        )
        assert {element.text for element in results} == {f"text-{index}" for index in range(4)}


@pytest.mark.asyncio
async def test_qdrant_vector_index_search_without_query_applies_requirements() -> None:
    async with _qdrant_scope():
        results: Sequence[_Document] = await VectorIndex.search(
            _Document,
            requirements=AttributeRequirement[_Document].equal(
                "odd",
                path=_Document._.tag,
            ),
            limit=8,
        )
        assert {element.text for element in results} == {"text-1", "text-3"}


@pytest.mark.asyncio
async def test_qdrant_create_payload_index_uses_supported_argument() -> None:
    async with _qdrant_scope():
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # local Qdrant reports payload indexes as ineffective, that is not our concern here
            warnings.simplefilter("ignore", UserWarning)
            assert await Qdrant.create_index(
                _Document,
                path=_Document._.tag,
                index_type="keyword",
            )


@pytest.mark.asyncio
async def test_qdrant_create_collection_supports_disk_placement() -> None:
    async with ctx.scope(
        "qdrant-test",
        disposables=[QdrantClient(in_memory=True)],
    ):
        assert await Qdrant.create_collection(
            _Document,
            vector_size=2,
            in_ram=False,
        )


@pytest.mark.asyncio
async def test_qdrant_contains_requirement_matches_array_elements() -> None:
    """Regression test: 'contains' used to raise NotImplementedError although a
    plain value match against an array payload field matches any of its elements.
    """
    async with _qdrant_scope():
        results: Sequence[_Document] = await VectorIndex.search(
            _Document,
            requirements=AttributeRequirement[_Document].contains(
                "tag-2",
                path=_Document._.tags,
            ),
            limit=8,
        )
        assert {element.text for element in results} == {"text-2"}


class _Measurement(State):
    label: str
    score: float
    identifier: UUID
    recorded: datetime


@asynccontextmanager
async def _measurement_scope() -> AsyncGenerator[Sequence[_Measurement]]:
    values: Sequence[_Measurement] = tuple(
        _Measurement(
            label=f"label-{index}",
            score=index / 2,
            identifier=UUID(int=index),
            recorded=datetime(2026, 1, index + 1, tzinfo=UTC),
        )
        for index in range(4)
    )
    async with ctx.scope(
        "qdrant-test",
        QdrantVectorIndex.prepare(),
        disposables=[QdrantClient(in_memory=True)],
    ):
        await Qdrant.create_collection(
            _Measurement,
            vector_size=2,
            in_ram=True,
        )
        await Qdrant.store(
            _Measurement,
            objects=[
                Embedded(
                    value=value,
                    vector=[float(index), 1.0],
                )
                for index, value in enumerate(values)
            ],
        )
        yield values


@pytest.mark.asyncio
async def test_qdrant_equal_requirement_matches_float_values() -> None:
    """Regression test: `MatchValue` accepts strictly bool, int and str - a float
    used to fail its validation instead of being matched through an exact range.
    """
    async with _measurement_scope():
        results: Sequence[_Measurement] = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].equal(
                1.5,
                path=_Measurement._.score,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-3"}


@pytest.mark.asyncio
async def test_qdrant_contained_in_requirement_matches_float_values() -> None:
    async with _measurement_scope():
        results: Sequence[_Measurement] = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].contained_in(
                (0.5, 1.5),
                path=_Measurement._.score,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-1", "label-3"}


@pytest.mark.asyncio
async def test_qdrant_integer_requirement_matches_float_payload() -> None:
    """Regression test: an integer requirement value used to match no float payload,
    silently returning nothing - and everything for its negation.
    """
    async with _measurement_scope():
        results: Sequence[_Measurement] = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].equal(
                1,
                path=_Measurement._.score,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-2"}

        results = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].not_equal(
                1,
                path=_Measurement._.score,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-0", "label-1", "label-3"}

        results = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].contained_in(
                (0, 1),
                path=_Measurement._.score,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-0", "label-2"}


@pytest.mark.asyncio
async def test_qdrant_equal_requirement_matches_uuid_and_datetime_values() -> None:
    """Regression test: UUID and datetime used to fail `MatchValue` validation,
    both are stored within the payload as their string representation.
    """
    async with _measurement_scope():
        results: Sequence[_Measurement] = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].equal(
                UUID(int=2),
                path=_Measurement._.identifier,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-2"}

        results = await VectorIndex.search(
            _Measurement,
            requirements=AttributeRequirement[_Measurement].equal(
                datetime(2026, 1, 2, tzinfo=UTC),
                path=_Measurement._.recorded,
            ),
            limit=8,
        )
        assert {element.label for element in results} == {"label-1"}


def test_qdrant_not_equal_requirement_matches_float_values() -> None:
    condition: Filter | None = prepare_filter(
        requirements=AttributeRequirement[_Measurement].not_equal(
            1.5,
            path=_Measurement._.score,
        )
    )
    assert condition is not None
    assert condition.must_not == [
        FieldCondition(
            key="score",
            range=Range(
                gte=1.5,
                lte=1.5,
            ),
        )
    ]


def test_qdrant_integer_requirement_uses_range() -> None:
    """Regression test: `MatchValue` matches an integer only against an integer
    payload, a float payload of an equal value used to match nothing at all.
    """
    condition: Filter | None = prepare_filter(
        requirements=AttributeRequirement[_Measurement].equal(
            2,
            path=_Measurement._.score,
        )
    )
    assert condition is not None
    assert condition.must == [
        FieldCondition(
            key="score",
            range=Range(
                gte=2,
                lte=2,
            ),
        )
    ]


def test_qdrant_integer_collection_requirement_uses_ranges() -> None:
    """Regression test: an all integer `MatchAny` never matched float payloads."""
    condition: Filter | None = prepare_filter(
        requirements=AttributeRequirement[_Measurement].contained_in(
            (1, 2),
            path=_Measurement._.score,
        )
    )
    assert condition is not None
    assert condition.should == [
        FieldCondition(
            key="score",
            range=Range(
                gte=1,
                lte=1,
            ),
        ),
        FieldCondition(
            key="score",
            range=Range(
                gte=2,
                lte=2,
            ),
        ),
    ]


def test_qdrant_keyword_collection_requirement_uses_match_any() -> None:
    condition: Filter | None = prepare_filter(
        requirements=AttributeRequirement[_Document].contained_in(
            ("even", "odd"),
            path=_Document._.tag,
        )
    )
    assert condition is not None
    assert condition.must == [
        FieldCondition(
            key="tag",
            match=MatchAny(any=["even", "odd"]),
        )
    ]


def test_qdrant_datetime_requirement_uses_datetime_range() -> None:
    """Regression test: payload datetimes are stored using Qdrant's own
    representation, an exact string match would never compare equal.
    """
    condition: Filter | None = prepare_filter(
        requirements=AttributeRequirement[_Measurement].equal(
            datetime(2026, 1, 2, tzinfo=UTC),
            path=_Measurement._.recorded,
        )
    )
    assert condition is not None
    assert condition.must == [
        FieldCondition(
            key="recorded",
            range=DatetimeRange(
                gte=datetime(2026, 1, 2, tzinfo=UTC),
                lte=datetime(2026, 1, 2, tzinfo=UTC),
            ),
        )
    ]


def test_qdrant_unsupported_requirement_value_is_reported() -> None:
    with pytest.raises(NotImplementedError):
        prepare_filter(
            requirements=AttributeRequirement[_Document].equal(
                object(),
                path=_Document._.tag,
            )
        )


@pytest.mark.asyncio
async def test_qdrant_unknown_extra_arguments_are_rejected() -> None:
    """Regression test: the client asserts on unrecognized keyword arguments,
    which used to surface as an AssertionError - and is skipped entirely when
    running with assertions disabled.
    """
    async with _qdrant_scope():
        with pytest.raises(ValueError, match="Unsupported Qdrant"):
            await Qdrant.fetch(
                _Document,
                unsupported_argument=True,
            )

        with pytest.raises(ValueError, match="Unsupported Qdrant"):
            await Qdrant.search(
                _Document,
                query_vector=[1.0, 1.0],
                unsupported_argument=True,
            )


@pytest.mark.asyncio
async def test_qdrant_missing_collection_errors_are_translated() -> None:
    """Regression test: client errors used to surface as raw gRPC/http exceptions."""

    class _Unprovisioned(State):
        text: str

    async with _qdrant_scope():
        with pytest.raises(QdrantException, match="fetching failed"):
            await Qdrant.fetch(_Unprovisioned)

        with pytest.raises(QdrantException, match="searching failed"):
            await Qdrant.search(
                _Unprovisioned,
                query_vector=[1.0, 1.0],
            )

        with pytest.raises(QdrantException, match="deleting failed"):
            await Qdrant.delete(_Unprovisioned)


@pytest.mark.asyncio
async def test_qdrant_store_errors_are_translated() -> None:
    async with _qdrant_scope():
        with pytest.raises(QdrantException, match="storing failed"):
            await Qdrant.store(
                _Document,
                objects=[
                    Embedded(
                        value=_Document(text="text", tag="even", tags=()),
                        # the collection is provisioned with a vector size of 2
                        vector=[1.0, 1.0, 1.0],
                    )
                ],
            )


@pytest.mark.asyncio
async def test_qdrant_create_collection_verifies_extra_when_existing() -> None:
    """Regression test: unrecognized arguments used to be dropped silently
    whenever the collection existed already.
    """
    async with _qdrant_scope():
        with pytest.raises(ValueError, match="Unsupported Qdrant"):
            await Qdrant.create_collection(
                _Document,
                vector_size=2,
                unsupported_argument=True,
            )


@pytest.mark.asyncio
async def test_qdrant_create_index_verifies_extra_when_missing() -> None:
    class _Unprovisioned(State):
        text: str

    async with _qdrant_scope():
        with pytest.raises(ValueError, match="Unsupported Qdrant"):
            await Qdrant.create_index(
                _Unprovisioned,
                path=_Unprovisioned._.text,
                index_type="keyword",
                unsupported_argument=True,
            )


class _AliasedNested(State):
    inner_value: Annotated[str, Alias("inner")]


class _AliasedDocument(State):
    identifier: Annotated[str, Alias("id")]
    text: str
    nested: _AliasedNested


def test_prepare_filter_uses_serialized_payload_keys() -> None:
    """Regression test: payloads are stored serialized, filtering by the python
    attribute name of an aliased field silently matched nothing.
    """
    prepared: Filter | None = prepare_filter(
        AttributeRequirement[_AliasedDocument].equal(
            "doc-1",
            path=_AliasedDocument._.identifier,
        )
    )

    assert prepared is not None
    condition = prepared.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "id"


def test_prepare_filter_resolves_nested_aliases() -> None:
    prepared: Filter | None = prepare_filter(
        AttributeRequirement[_AliasedDocument].equal(
            "value",
            path=_AliasedDocument._.nested.inner_value,
        )
    )

    assert prepared is not None
    condition = prepared.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "nested.inner"


@asynccontextmanager
async def _aliased_scope() -> AsyncGenerator[None]:
    async with ctx.scope(
        "qdrant-test",
        QdrantVectorIndex.prepare(),
        disposables=[QdrantClient(in_memory=True)],
    ):
        await Qdrant.create_collection(_AliasedDocument, vector_size=2, in_ram=True)
        await Qdrant.store(
            _AliasedDocument,
            objects=[
                Embedded(
                    value=_AliasedDocument(
                        identifier=f"doc-{index}",
                        text=f"text-{index}",
                        nested=_AliasedNested(inner_value=f"inner-{index}"),
                    ),
                    vector=[float(index), 1.0],
                )
                for index in range(3)
            ],
        )
        yield


@pytest.mark.asyncio
async def test_qdrant_search_filters_by_aliased_attribute() -> None:
    async with _aliased_scope():
        results: Sequence[_AliasedDocument] = await VectorIndex.search(
            _AliasedDocument,
            requirements=AttributeRequirement[_AliasedDocument].equal(
                "doc-2",
                path=_AliasedDocument._.identifier,
            ),
            limit=8,
        )

        assert [element.identifier for element in results] == ["doc-2"]
