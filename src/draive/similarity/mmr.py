from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine

__all__ = [
    "mmr_similarity",
]


def mmr_similarity(
    query_embedding: NDArray[Any] | list[float],
    alternatives_embeddings: list[NDArray[Any]] | list[list[float]],
    limit: int,
    lambda_multiplier: float = 0.5,
) -> list[int]:
    assert limit > 0  # nosec: B101
    if not alternatives_embeddings:
        return []

    query: NDArray[Any] = np.array(query_embedding)
    if query.ndim == 1:
        query = np.expand_dims(query_embedding, axis=0)
    alternatives: NDArray[Any] = np.array(alternatives_embeddings)

    # count similarity
    similarity: NDArray[Any] = cosine(alternatives, query)
    # find most similar match for query
    most_similar: int = int(np.argmax(similarity))
    selected_indices: list[int] = [most_similar]
    selected: NDArray[Any] = np.array([alternatives[most_similar]])

    # then look one by one next best matches until the limit or end of alternatives
    while len(selected_indices) < limit and len(selected_indices) < len(alternatives_embeddings):
        best_score: float = -np.inf
        best_index: int = -1
        # count similarity to already selected results
        similarity_to_selected: NDArray[Any] = cosine(alternatives, selected)

        # then find the next best score
        # (balancing between similarity to query and uniqueness of result)
        for idx, similarity_score in enumerate(similarity):
            if idx in selected_indices:
                continue  # skip already added

            equation_score = (
                lambda_multiplier * similarity_score
                - (1 - lambda_multiplier) * similarity_to_selected[idx]
            )
            # check if has better score
            if equation_score > best_score:
                best_score = equation_score
                best_index = idx

        if best_index < 0:
            break

        selected_indices.append(best_index)
        selected = np.append(
            selected,
            [alternatives[best_index]],  # pyright: ignore[reportUnknownArgumentType]
            axis=0,
        )

    return selected_indices
