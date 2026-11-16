from collections.abc import Callable, Sequence
from typing import Any

import pytest
from haiway import AttributeRequirement, State, ctx

from draive.embedding import Embedded, TextEmbedding, VectorIndex
from draive.multimodal import TextContent
from draive.qdrant import Qdrant, QdrantClient, QdrantException
from draive.qdrant.index import QdrantVectorIndex
from draive.resources import ResourceContent


class _Note(State):
    text: str
    group: str


_VECTORS: dict[str, tuple[float, float]] = {
    "alpha": (1.0, 0.0),
    "beta": (0.0, 1.0),
    "gamma": (0.9, 0.1),
}


async def _embedding(
    values: Sequence[str],
    /,
    attribute: Callable[[str], str] | None = None,
    **extra: Any,
) -> Sequence[Embedded[str]]:
    return [Embedded(value=value, vector=_VECTORS[value]) for value in values]


def _scope() -> Any:
    return ctx.scope(
        "qdrant-flow",
        QdrantVectorIndex.prepare(),
        TextEmbedding(embedding=_embedding),
        disposables=[QdrantClient(in_memory=True)],
    )


async def _populate() -> None:
    await Qdrant.create_collection(_Note, vector_size=2, in_ram=True)
    await VectorIndex.index(
        _Note,
        attribute=_Note._.text,
        values=[
            _Note(text="alpha", group="a"),
            _Note(text="beta", group="b"),
            _Note(text="gamma", group="a"),
        ],
    )


@pytest.mark.asyncio
async def test_index_and_search_by_text_query_orders_by_similarity() -> None:
    async with _scope():
        await _populate()
        results = await VectorIndex.search(_Note, query="alpha", limit=2)

    assert [note.text for note in results] == ["alpha", "gamma"]


@pytest.mark.asyncio
async def test_search_accepts_text_content_and_raw_vector_queries() -> None:
    async with _scope():
        await _populate()
        by_content = await VectorIndex.search(_Note, query=TextContent.of("beta"), limit=1)
        by_vector = await VectorIndex.search(_Note, query=[0.0, 1.0], limit=1)

    assert [note.text for note in by_content] == ["beta"]
    assert [note.text for note in by_vector] == ["beta"]


@pytest.mark.asyncio
async def test_search_applies_score_threshold_and_requirements() -> None:
    async with _scope():
        await _populate()
        thresholded = await VectorIndex.search(
            _Note,
            query="alpha",
            score_threshold=0.999,
            limit=8,
        )
        filtered = await VectorIndex.search(
            _Note,
            query="alpha",
            requirements=AttributeRequirement[_Note].equal("b", path=_Note._.group),
            limit=8,
        )

    assert [note.text for note in thresholded] == ["alpha"]
    assert [note.text for note in filtered] == ["beta"]


@pytest.mark.asyncio
async def test_rerank_returns_requested_number_of_results() -> None:
    async with _scope():
        await _populate()
        results = await VectorIndex.search(_Note, query="alpha", limit=2, rerank=True)

    assert len(results) == 2
    assert results[0].text == "alpha"


@pytest.mark.asyncio
async def test_rerank_on_empty_collection_returns_nothing() -> None:
    async with _scope():
        await Qdrant.create_collection(_Note, vector_size=2, in_ram=True)
        results = await VectorIndex.search(_Note, query=[1.0, 0.0], rerank=True, limit=3)

    assert results == ()


@pytest.mark.asyncio
async def test_delete_with_requirements_removes_only_matching() -> None:
    async with _scope():
        await _populate()
        await VectorIndex.delete(
            _Note,
            requirements=AttributeRequirement[_Note].equal("a", path=_Note._.group),
        )
        remaining = await VectorIndex.search(_Note, limit=8)

        await VectorIndex.delete(_Note)
        emptied = await VectorIndex.search(_Note, limit=8)

    assert [note.text for note in remaining] == ["beta"]
    assert emptied == ()


@pytest.mark.asyncio
async def test_index_without_values_does_not_embed_or_store() -> None:
    async with _scope():
        await Qdrant.create_collection(_Note, vector_size=2, in_ram=True)
        await VectorIndex.index(_Note, attribute=_Note._.text, values=[])
        assert await VectorIndex.search(_Note, limit=8) == ()


@pytest.mark.asyncio
async def test_non_image_resource_embedding_is_rejected() -> None:
    document = ResourceContent.of(b"%PDF", mime_type="application/pdf")

    async with _scope():
        await Qdrant.create_collection(_Note, vector_size=2, in_ram=True)
        with pytest.raises(ValueError, match="not supported"):
            await VectorIndex.search(_Note, query=document)


@pytest.mark.asyncio
async def test_search_with_mismatched_vector_size_is_translated() -> None:
    async with _scope():
        await _populate()
        with pytest.raises(QdrantException):
            await VectorIndex.search(_Note, query=[1.0], limit=2)


def test_invalid_mmr_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="mmr_multiplier"):
        QdrantVectorIndex.prepare(mmr_multiplier=0)


@pytest.mark.asyncio
async def test_search_with_zero_limit_returns_nothing() -> None:
    async with _scope():
        await _populate()
        results = await VectorIndex.search(_Note, query="alpha", limit=0)

    assert results == ()


@pytest.mark.asyncio
async def test_search_without_query_and_zero_limit_returns_nothing() -> None:
    async with _scope():
        await _populate()
        results = await VectorIndex.search(_Note, limit=0)

    assert results == ()
