from typing import Any

import numpy as np
from numpy.typing import NDArray

from draive.similarity.cosine import cosine

__all__ = [
    "similarity_score",
]


async def similarity_score(
    value_vector: NDArray[Any] | list[float],
    reference_vector: NDArray[Any] | list[float],
) -> float:
    reference: NDArray[Any] = np.array(reference_vector)
    if reference.ndim == 1:
        reference = np.expand_dims(reference_vector, axis=0)
    value: NDArray[Any] = np.array(value_vector)
    if value.ndim == 1:
        value = np.expand_dims(value_vector, axis=0)

    return cosine(value, reference)[0]
