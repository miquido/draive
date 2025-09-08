from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ("cosine_similarity",)

# Expected dimensionality for 2D matrices (n x d)
_EXPECTED_NDIM: int = 2


def cosine_similarity(
    a: Sequence[NDArray[Any]] | NDArray[Any],
    b: Sequence[NDArray[Any]] | NDArray[Any],
) -> NDArray[Any]:
    """Compute pairwise cosine similarity between vectors.

    Parameters
    ----------
    a
        A 1D vector, a 2D array of shape (n, d), or a sequence of 1D vectors.
    b
        A 1D vector, a 2D array of shape (m, d), or a sequence of 1D vectors.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the flattened (row-major) pairwise cosine
        similarity matrix of shape (n * m,).

    Raises
    ------
    ValueError
        If vectors in inputs do not all share the same dimensionality, or
        input elements have incompatible shapes and cannot be converted into
        a 2D array of vectors.
    """

    def _to_2d(arr: NDArray[Any] | Sequence[Any]) -> NDArray[Any]:
        """Coerce a vector or sequence of vectors into a 2D array.

        Accepts a 1D vector (d,), a 2D array (n, d) or a sequence of 1D vectors,
        and returns a numeric array of shape (n, d).
        """
        arr = np.asarray(arr)

        # If it's an object array (likely ragged or a sequence of arrays), try stacking
        if arr.dtype == np.dtype(object):
            try:
                arr = np.stack([np.asarray(v) for v in arr], axis=0)
            except Exception as exc:  # numpy raises ValueError on mismatch
                raise ValueError("Input vectors must all have the same length and shape.") from exc

        # Ensure we have a 2D array (1 x d or n x d)
        arr = np.atleast_2d(arr)
        if arr.ndim != _EXPECTED_NDIM:
            raise ValueError("Input must be a 1D vector or a 2D array of vectors.")
        return arr

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.size == 0 or b_arr.size == 0:
        return np.array([])

    x: NDArray[Any] = _to_2d(a_arr)
    y: NDArray[Any] = _to_2d(b_arr)

    if x.shape[1] != y.shape[1]:
        raise ValueError("Vector dimensionality must match for both arguments.")

    # Normalize rows safely to avoid NaNs/Infs for zero vectors
    x_norms: NDArray[Any] = np.linalg.norm(x, axis=1)
    y_norms: NDArray[Any] = np.linalg.norm(y, axis=1)

    # Replace zeros with ones to keep zero vectors as zero after division
    x_safe: NDArray[Any] = x / np.where(x_norms == 0, 1.0, x_norms)[:, None]
    y_safe: NDArray[Any] = y / np.where(y_norms == 0, 1.0, y_norms)[:, None]

    similarity: NDArray[Any] = x_safe @ y_safe.T

    return similarity.flatten()
