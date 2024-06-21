from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine_similarity

__all__ = [
    "vector_similarity_score",
]


def vector_similarity_score(
    value_vector: NDArray[Any] | list[float],
    reference_vector: NDArray[Any] | list[float],
    similarity: Callable[
        [list[NDArray[Any]] | NDArray[Any], list[NDArray[Any]] | NDArray[Any]], NDArray[Any]
    ] = cosine_similarity,
) -> float:
    reference: NDArray[Any] = np.array(reference_vector)
    if reference.ndim == 1:
        reference = np.expand_dims(reference_vector, axis=0)

    value: NDArray[Any] = np.array(value_vector)
    if value.ndim == 1:
        value = np.expand_dims(value_vector, axis=0)

    return similarity(value, reference)[0]
