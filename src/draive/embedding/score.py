from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.embedding.cosine import cosine_similarity

__all__ = ("vector_similarity_score",)


def vector_similarity_score(
    value_vector: NDArray[Any] | Sequence[float],
    reference_vector: NDArray[Any] | Sequence[float],
    similarity: Callable[
        [list[NDArray[Any]] | NDArray[Any], list[NDArray[Any]] | NDArray[Any]], NDArray[Any]
    ] = cosine_similarity,
) -> float:
    reference: NDArray[Any] = np.array(reference_vector)
    if reference.ndim == 1:
        reference = np.expand_dims(reference, axis=0)

    value: NDArray[Any] = np.array(value_vector)
    if value.ndim == 1:
        value = np.expand_dims(value, axis=0)

    sim: NDArray[Any] = np.array(similarity(value, reference)).reshape(-1)
    return float(sim[0])
