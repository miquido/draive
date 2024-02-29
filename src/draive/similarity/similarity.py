from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine

__all__ = [
    "similarity",
]


def similarity(
    query_embedding: NDArray[Any] | list[float],
    alternatives_embeddings: list[NDArray[Any]] | list[list[float]],
    limit: int,
    score_threshold: float,
) -> list[int]:
    assert limit > 0  # nosec: B101
    if not alternatives_embeddings:
        return []
    query: NDArray[Any] = np.array(query_embedding)
    if query.ndim == 1:
        query = np.expand_dims(query_embedding, axis=0)
    alternatives: NDArray[Any] = np.array(alternatives_embeddings)
    matching_scores: NDArray[Any] = cosine(alternatives, query)
    sorted_indices: list[int] = list(reversed(np.argsort(matching_scores)))
    return [
        int(idx)
        for idx in sorted_indices  # pyright: ignore[reportUnknownVariableType]
        if matching_scores[idx] > score_threshold  # pyright: ignore[reportUnknownArgumentType]
    ][:limit]
