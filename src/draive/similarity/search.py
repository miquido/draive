from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine

__all__ = [
    "similarity_search",
]


def similarity_search(
    query_vector: NDArray[Any] | list[float],
    values_vectors: list[NDArray[Any]] | list[list[float]],
    limit: int,
    score_threshold: float | None = None,
) -> list[int]:
    assert limit > 0  # nosec: B101
    if not values_vectors:
        return []
    query: NDArray[Any] = np.array(query_vector)
    if query.ndim == 1:
        query = np.expand_dims(query_vector, axis=0)
    values: NDArray[Any] = np.array(values_vectors)
    matching_scores: NDArray[Any] = cosine(values, query)
    sorted_indices: list[int] = list(reversed(np.argsort(matching_scores)))
    if score_threshold:
        return [
            int(idx)
            for idx in sorted_indices  # pyright: ignore[reportUnknownVariableType]
        ][:limit]
    else:
        return [
            int(idx)
            for idx in sorted_indices  # pyright: ignore[reportUnknownVariableType]
            if matching_scores[idx] > score_threshold  # pyright: ignore[reportUnknownArgumentType]
        ][:limit]
