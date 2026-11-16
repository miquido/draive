import math

import numpy as np
import pytest
from haiway import State, ctx

from draive.embedding import Embedded, TextEmbedding, VectorIndex
from draive.embedding.cosine import cosine_similarity
from draive.embedding.mmr import mmr_vector_similarity_search
from draive.embedding.score import vector_similarity_score
from draive.embedding.search import vector_similarity_search
from draive.helpers import VolatileVectorIndex

_NAN: float = float("nan")


def test_cosine_similarity_zero_vectors_do_not_produce_nan() -> None:
    result = cosine_similarity([[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0]])

    assert not np.isnan(result).any()
    assert result.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "call",
    (
        lambda: cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]),
        lambda: vector_similarity_search([1.0, 0.0, 0.0], [[1.0, 0.0]]),
        lambda: vector_similarity_score([1.0, 0.0, 0.0], [1.0, 0.0]),
        lambda: mmr_vector_similarity_search([1.0, 0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
    ),
    ids=("cosine", "search", "score", "mmr"),
)
def test_mismatched_dimensions_are_rejected(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_cosine_similarity_rejects_higher_dimensional_input() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros((2, 2, 2)), [1.0, 0.0])


def test_search_with_zero_limit_returns_nothing() -> None:
    assert vector_similarity_search([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], limit=0) == []


def test_search_with_threshold_and_limit_filters_before_truncating() -> None:
    result = vector_similarity_search(
        [1.0, 0.0],
        [[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]],
        score_threshold=0.5,
        limit=5,
    )

    assert list(result) == [1, 2]


def test_mmr_prefers_diverse_results_over_duplicates() -> None:
    result = mmr_vector_similarity_search(
        [1.0, 0.0],
        [[1.0, 0.0], [1.0, 0.0], [0.7, 0.7]],
        limit=2,
        lambda_multiplier=0.3,
    )

    assert list(result) == [0, 2]


def test_mmr_with_zero_vectors_returns_all_indices() -> None:
    result = mmr_vector_similarity_search([0.0, 0.0], [[0.0, 0.0]] * 3)

    assert sorted(result) == [0, 1, 2]


def test_mmr_rejects_lambda_out_of_range() -> None:
    with pytest.raises(ValueError):
        mmr_vector_similarity_search([1.0], [[1.0]], lambda_multiplier=1.5)


@pytest.mark.parametrize("limit", (-1,))
def test_search_with_negative_limit_returns_nothing(limit: int) -> None:
    assert vector_similarity_search([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], limit=limit) == []


def test_search_does_not_rank_nan_vectors_first() -> None:
    result = vector_similarity_search([1.0, 0.0], [[_NAN, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert result[0] == 1


def test_mmr_tolerates_nan_vectors() -> None:
    result = mmr_vector_similarity_search([1.0, 0.0], [[_NAN, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert result[0] == 1
    assert len(result) == 3


def test_similarity_score_of_empty_vectors_is_defined() -> None:
    score = vector_similarity_score([], [])

    assert math.isfinite(score)


class _Document(State):
    text: str


def _embedding() -> TextEmbedding:
    async def embedding(values, /, attribute=None, **extra):
        return [Embedded(value=value, vector=(1.0, 0.0)) for value in values]

    return TextEmbedding(embedding=embedding)


@pytest.mark.asyncio
async def test_volatile_vector_index_negative_limit_with_query_returns_nothing() -> None:
    async with ctx.scope("test", VolatileVectorIndex(), _embedding()):
        await VectorIndex.index(
            _Document,
            attribute=_Document._.text,
            values=(_Document(text="a"), _Document(text="b")),
        )

        results = await VectorIndex.search(_Document, query=(1.0, 0.0), limit=-1)

    assert list(results) == []


def test_search_with_nan_vectors_keeps_them_last() -> None:
    result = vector_similarity_search([1.0, 0.0], [[_NAN, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert list(result) == [1, 2, 0]


def test_search_with_nan_vectors_and_limit_prefers_real_scores() -> None:
    result = vector_similarity_search(
        [1.0, 0.0],
        [[_NAN, 0.0], [1.0, 0.0], [0.0, 1.0]],
        limit=2,
    )

    assert list(result) == [1, 2]


@pytest.mark.parametrize(
    ("a", "b"),
    (
        ([[math.inf, 0.0]], [[1.0, 0.0]]),
        ([[1.0, 0.0]], [[0.0, -math.inf]]),
        ([[math.inf, math.inf]], [[math.inf, 1.0]]),
    ),
)
def test_cosine_similarity_infinite_vectors_never_leak_nan(a, b) -> None:
    result = cosine_similarity(a, b)

    assert not np.isnan(result).any()
    assert np.isfinite(result).all()
    assert ((result >= -1.0) & (result <= 1.0)).all()
