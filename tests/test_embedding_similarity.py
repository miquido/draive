import math

import numpy as np
import pytest

from draive.embedding.cosine import cosine_similarity
from draive.embedding.mmr import mmr_vector_similarity_search
from draive.embedding.score import vector_similarity_score
from draive.embedding.search import vector_similarity_search


def test_cosine_similarity_identity_and_orthogonal_1d_and_2d() -> None:
    v = np.array([1.0, 0.0])
    u = np.array([0.0, 1.0])

    # 1D inputs
    sim_1d = cosine_similarity(v, v)
    assert sim_1d.shape == (1,)
    assert math.isclose(float(sim_1d[0]), 1.0, rel_tol=1e-9)

    sim_orth_1d = cosine_similarity(v, u)
    assert sim_orth_1d.shape == (1,)
    assert math.isclose(float(sim_orth_1d[0]), 0.0, abs_tol=1e-12)

    # 2D inputs (pairwise matrix, flattened)
    sim_2d = cosine_similarity(np.vstack([v, u]), np.vstack([v, u]))
    # Values should be: [1, 0, 0, 1] in some flattened order
    assert set(np.round(sim_2d, 6).tolist()) == {0.0, 1.0}


def test_cosine_similarity_zero_vector_safe() -> None:
    zero = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    sim = cosine_similarity(zero, v)
    assert sim.shape == (1,)
    assert math.isclose(float(sim[0]), 0.0, abs_tol=1e-12)


def test_cosine_similarity_handles_sequence_of_vectors_and_vector() -> None:
    a = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    b = np.array([1.0, 0.0])
    sim = cosine_similarity(a, b)
    assert sim.shape == (2,)
    assert np.allclose(sim, np.array([1.0, 0.0]), atol=1e-12)


def test_cosine_similarity_ragged_incompatible_raises() -> None:
    a = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0])]
    b = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        _ = cosine_similarity(a, b)


def test_cosine_similarity_empty_input_returns_empty() -> None:
    assert cosine_similarity([], [1.0, 0.0]).size == 0


def test_vector_similarity_search_order_and_limit() -> None:
    # query closest to v0, then v1, then v2
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.9, 0.1])
    v2 = np.array([0.0, 1.0])
    values = np.array([v0, v1, v2])
    query = np.array([1.0, 0.0])

    order = vector_similarity_search(query, values, limit=3)
    assert order == [0, 1, 2]

    top1 = vector_similarity_search(query, values, limit=1)
    assert top1 == [0]

    # top-2 using efficient path should return [0, 1]
    top2 = vector_similarity_search(query, values, limit=2)
    assert top2 == [0, 1]


def test_vector_similarity_search_handles_single_1d_values_vector() -> None:
    query = np.array([1.0, 0.0])
    values_1d = np.array([1.0, 0.0])
    order = vector_similarity_search(query, values_1d, limit=3)
    assert order == [0]

    # also accept Python list as single vector
    order_list = vector_similarity_search(query, [1.0, 0.0], limit=3)
    assert order_list == [0]


def test_vector_similarity_search_threshold_applied_then_limit() -> None:
    # Construct vectors to yield descending similarities ~ [1.0, 0.8, 0.7, 0.0]
    query = np.array([1.0, 0.0])
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.8, 0.6])  # normalized; dot with query is 0.8
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array([0.7, math.sqrt(1 - 0.7**2)])  # normalized; dot 0.7
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.array([0.0, 1.0])
    values = np.array([v0, v1, v2, v3])

    # Threshold 0.75 keeps only indices 0 and 1; limit=1 keeps just top
    result = vector_similarity_search(query, values, limit=1, score_threshold=0.75)
    assert result == [0]

    # With larger limit, both above-threshold items are returned
    result2 = vector_similarity_search(query, values, limit=3, score_threshold=0.75)
    assert result2 == [0, 1]


def test_vector_similarity_score_type_and_value() -> None:
    # Mixed list/ndarray inputs; identical direction -> score 1.0
    value = [1.0, 0.0]
    reference = np.array([2.0, 0.0])
    score = vector_similarity_score(value, reference)
    assert isinstance(score, float)
    assert math.isclose(score, 1.0, rel_tol=1e-9)


def test_mmr_promotes_diversity_with_low_lambda() -> None:
    # v0 and v1 are very similar; v2 is diverse; low lambda favors diversity
    v0 = np.array([1.0, 0.0])
    v1 = np.array([0.99, 0.01])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array([0.0, 1.0])
    values = np.array([v0, v1, v2])
    query = np.array([1.0, 0.0])

    order = mmr_vector_similarity_search(query, values, limit=2, lambda_multiplier=0.1)
    assert order == [0, 2]


def test_search_and_mmr_handle_empty_values() -> None:
    assert vector_similarity_search([1.0, 0.0], []) == []
    assert mmr_vector_similarity_search([1.0, 0.0], []) == []
