from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine_similarity

__all__ = [
    "mmr_vector_similarity_search",
]


def mmr_vector_similarity_search(
    query_vector: NDArray[Any] | list[float],
    values_vectors: list[NDArray[Any]] | list[list[float]],
    limit: int,
    lambda_multiplier: float = 0.5,
    similarity: Callable[
        [list[NDArray[Any]] | NDArray[Any], list[NDArray[Any]] | NDArray[Any]], NDArray[Any]
    ] = cosine_similarity,
) -> list[int]:
    assert limit > 0  # nosec: B101
    if not values_vectors:
        return []

    query: NDArray[Any] = np.array(query_vector)
    if query.ndim == 1:
        query = np.expand_dims(query_vector, axis=0)
    values: NDArray[Any] = np.array(values_vectors)

    # count similarity
    current_similarity: NDArray[Any] = similarity(values, query)
    # find most similar match for query
    most_similar: int = int(np.argmax(current_similarity))
    selected_indices: list[int] = [most_similar]
    selected: NDArray[Any] = np.array([values[most_similar]])

    # then look one by one next best matches until the limit or end of alternatives
    while len(selected_indices) < limit and len(selected_indices) < len(values_vectors):
        best_score: float = -np.inf
        best_index: int = -1
        # count similarity to already selected results
        similarity_to_selected: NDArray[Any] = similarity(values, selected)

        # then find the next best score
        # (balancing between similarity to query and uniqueness of result)
        for idx, similarity_score in enumerate(current_similarity):
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
            [values[best_index]],  # pyright: ignore[reportUnknownArgumentType]
            axis=0,
        )

    return selected_indices
