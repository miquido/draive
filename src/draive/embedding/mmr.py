from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.embedding.cosine import cosine_similarity

__all__ = ("mmr_vector_similarity_search",)


def mmr_vector_similarity_search(  # noqa: C901
    query_vector: NDArray[Any] | Sequence[float],
    values_vectors: Sequence[NDArray[Any]] | Sequence[Sequence[float]],
    limit: int | None = None,
    lambda_multiplier: float = 0.5,
    similarity: Callable[
        [list[NDArray[Any]] | NDArray[Any], list[NDArray[Any]] | NDArray[Any]], NDArray[Any]
    ] = cosine_similarity,
) -> Sequence[int]:
    """Select indices using Maximal Marginal Relevance (MMR).

    Balances relevance to the query with diversity among selected results. The
    ``lambda_multiplier`` controls the trade-off: 1.0 favors only similarity to
    the query, 0.0 favors maximal dissimilarity to items already selected.

    Parameters
    ----------
    query_vector
        Query embedding vector.
    values_vectors
        Sequence of embedding vectors to search through.
    limit
        Maximum number of indices to return. Defaults to the number of input
        vectors when not provided.
    lambda_multiplier
        Trade-off factor in [0.0, 1.0] balancing query similarity and
        diversity. Higher values prioritize similarity to the query.
    similarity
        Pairwise similarity function returning a flattened similarity array.

    Returns
    -------
    Sequence[int]
        Indices of selected vectors ordered by MMR ranking.

    Raises
    ------
    ValueError
        If ``lambda_multiplier`` is outside the inclusive range [0.0, 1.0].
    """
    if len(values_vectors) == 0:
        return []

    # Validate lambda multiplier range per contract
    if not (0.0 <= lambda_multiplier <= 1.0):
        raise ValueError("lambda_multiplier must be within [0.0, 1.0]")

    results_limit: int
    if limit is not None:
        results_limit = min(limit, len(values_vectors))

    else:
        results_limit = len(values_vectors)

    query: NDArray[Any] = np.array(query_vector)
    if query.ndim == 1:
        # ensure 2D shape for similarity calculation
        query = np.expand_dims(query, axis=0)
    values: NDArray[Any] = np.array(values_vectors)

    # count similarity
    current_similarity: NDArray[Any] = np.array(similarity(values, query)).reshape(-1)
    # find most similar match for query
    most_similar: int = int(np.argmax(current_similarity))
    selected_indices: list[int] = [most_similar]
    selected: NDArray[Any] = np.array([values[most_similar]])

    # then look one by one next best matches until the limit
    while len(selected_indices) < results_limit:
        best_score: float = -np.inf
        best_index: int = -1
        # count similarity to already selected results
        similarity_to_selected: NDArray[Any] = similarity(values, selected)
        # similarity may be returned flattened (N * K); reshape to (N, K)
        if similarity_to_selected.ndim == 1:
            similarity_to_selected = similarity_to_selected.reshape(
                values.shape[0],
                selected.shape[0],
            )

        # then find the next best score
        # (balancing between similarity to query and uniqueness of result)
        for idx, similarity_score in enumerate(current_similarity):
            if idx in selected_indices:
                continue  # skip already added

            max_similarity_to_selected: float = float(np.max(similarity_to_selected[idx]))
            equation_score = (
                lambda_multiplier * float(similarity_score)
                - (1 - lambda_multiplier) * max_similarity_to_selected
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
