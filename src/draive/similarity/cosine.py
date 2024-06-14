from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "cosine_similarity",
]


def cosine_similarity(
    a: list[NDArray[Any]] | NDArray[Any],
    b: list[NDArray[Any]] | NDArray[Any],
) -> NDArray[Any]:
    if len(a) == 0 or len(b) == 0:
        return np.array([])

    x: NDArray[Any] = np.array(a)
    y: NDArray[Any] = np.array(b)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Number of columns has to be the same for both arguments.")

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity: NDArray[Any] = np.dot(
            x,
            y.T,
        ) / np.outer(
            np.linalg.norm(x, axis=1),
            np.linalg.norm(y, axis=1),
        )

    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0

    return similarity.flatten()
