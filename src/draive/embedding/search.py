from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.embedding.cosine import cosine_similarity

__all__ = ("vector_similarity_search",)


def vector_similarity_search(
    query_vector: NDArray[Any] | Sequence[float],
    values_vectors: Sequence[NDArray[Any]] | Sequence[Sequence[float]],
    limit: int | None = None,
    score_threshold: float | None = None,
    similarity: Callable[
        [list[NDArray[Any]] | NDArray[Any], list[NDArray[Any]] | NDArray[Any]], NDArray[Any]
    ] = cosine_similarity,
) -> Sequence[int]:
    """Return indices of the most similar vectors to a query.

    Parameters
    ----------
    query_vector
        The query vector as a 1D array-like or a 2D array with shape (1, d).
    values_vectors
        Collection of vectors to search, as a 2D array of shape (n, d) or a
        sequence of 1D vectors. A single 1D vector is accepted and treated as
        one row.
    limit
        Optional maximum number of indices to return. If provided and smaller
        than the number of vectors, a more efficient top-k selection is used.
    score_threshold
        Optional lower bound for similarity score; items with scores not
        meeting this threshold are filtered out.
    similarity
        Pairwise similarity function that accepts two vector collections and
        returns a 1D flattened array of scores.

    Returns
    -------
    Sequence[int]
        Indices of `values_vectors` ordered by descending similarity to the
        query, possibly filtered by `score_threshold` and truncated by `limit`.
    """
    if len(values_vectors) == 0:
        return []

    # Coerce to numeric arrays and ensure 2D shape (n x d)
    query: NDArray[Any] = np.atleast_2d(np.asarray(query_vector, dtype=float))

    values: NDArray[Any] = np.asarray(values_vectors, dtype=float)
    if values.ndim == 1:
        # handle single 1D vector provided as values
        values = np.atleast_2d(values)
    matching_scores: NDArray[Any] = np.array(similarity(values, query)).reshape(-1)
    matching_scores = matching_scores.astype(float, copy=False)

    n_scores = int(matching_scores.shape[0])
    if limit is not None and 0 < int(limit) < n_scores:
        k = int(limit)
        # Select top-k indices using argpartition (largest k elements)
        candidate_idx: NDArray[np.int_] = np.argpartition(matching_scores, -k)[-k:]
        # Order candidates by score descending
        order_within: NDArray[np.int_] = np.argsort(matching_scores[candidate_idx])[::-1]
        ordered_indices_arr: NDArray[np.int_] = candidate_idx[order_within]
    else:
        # Fallback to full sort
        ordered_indices_arr = np.argsort(matching_scores)[::-1]

    if score_threshold is not None:
        thr = float(score_threshold)
        ordered_indices_arr = np.array(
            [idx for idx in ordered_indices_arr if matching_scores[int(idx)] >= thr], dtype=int
        )

    if limit is not None:
        ordered_indices_arr = ordered_indices_arr[: int(limit)]

    return [int(idx) for idx in ordered_indices_arr]
