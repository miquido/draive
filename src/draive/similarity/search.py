from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine_similarity

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
    if not values_vectors:
        return []

    query: NDArray[Any] = np.array(query_vector)
    if query.ndim == 1:
        query = np.expand_dims(query_vector, axis=0)
    values: NDArray[Any] = np.array(values_vectors)
    matching_scores: NDArray[Any] = similarity(values, query)
    sorted_indices: list[int] = list(reversed(np.argsort(matching_scores)))
    if limit is not None:
        sorted_indices = sorted_indices[:limit]

    if score_threshold:
        return [
            int(idx)
            for idx in sorted_indices  # pyright: ignore[reportUnknownVariableType]
            if matching_scores[idx] > score_threshold  # pyright: ignore[reportUnknownArgumentType]
        ]

    else:
        return [
            int(idx)
            for idx in sorted_indices  # pyright: ignore[reportUnknownVariableType]
        ]
