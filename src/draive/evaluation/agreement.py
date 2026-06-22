from collections.abc import Callable, Sequence
from typing import Final

from draive.evaluation.value import (
    EVALUATION_SCORE_LEVELS,
    EvaluationScoreLevel,
    evaluation_score_value,
)

__all__ = (
    "cohen_kappa",
    "quadratic_weighted_kappa",
    "quantize_score",
)


# Canonical float value of each named level, derived from the single source of
# truth in draive.evaluation.value: (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
_LEVEL_VALUES: Final[tuple[float, ...]] = tuple(
    evaluation_score_value(level) for level in EVALUATION_SCORE_LEVELS
)


def quantize_score(
    value: float,
    /,
) -> EvaluationScoreLevel:
    """
    Map a continuous score in [0, 1] to the nearest named evaluation level.

    Parameters
    ----------
    value : float
        Score value between 0.0 and 1.0.

    Returns
    -------
    EvaluationScoreLevel
        Name of the closest level (one of ``EVALUATION_SCORE_LEVELS``).

    Raises
    ------
    ValueError
        If the value is outside the [0, 1] range.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Score outside [0, 1]: {value}")

    # On an exact tie between two adjacent levels (e.g. 0.1, 0.3) ``min`` keeps the
    # first match, biasing toward the lower level - a deliberate, stable choice.
    closest_index: int = min(
        range(len(_LEVEL_VALUES)),
        key=lambda index: abs(value - _LEVEL_VALUES[index]),
    )
    return EVALUATION_SCORE_LEVELS[closest_index]


def _confusion(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    categories: Sequence[str],
    /,
) -> list[list[int]]:
    index: dict[str, int] = {label: position for position, label in enumerate(categories)}
    size: int = len(categories)
    matrix: list[list[int]] = [[0] * size for _ in range(size)]

    for left, right in zip(rater_a, rater_b, strict=True):
        if left not in index:
            raise ValueError(f"Unknown category in first sequence: {left}")

        if right not in index:
            raise ValueError(f"Unknown category in second sequence: {right}")

        matrix[index[left]][index[right]] += 1

    return matrix


def _weighted_kappa(
    matrix: list[list[int]],
    weight: Callable[[int, int], float],
    /,
) -> float:
    # General Cohen's kappa: 1 - sum(w * observed) / sum(w * expected), where the
    # weight encodes how much each (i, j) disagreement counts. Nominal kappa uses a
    # 0/1 weight; the ordinal variant uses squared distance (see the callers below).
    size: int = len(matrix)
    total: int = sum(sum(row) for row in matrix)
    row_totals: list[int] = [sum(row) for row in matrix]
    col_totals: list[int] = [sum(matrix[r][c] for r in range(size)) for c in range(size)]

    observed: float = 0.0
    expected: float = 0.0
    for i in range(size):
        for j in range(size):
            w: float = weight(i, j)
            observed += w * matrix[i][j]
            expected += w * (row_totals[i] * col_totals[j]) / total

    if expected == 0.0:
        # No disagreement is expected by chance (a single populated category):
        # the result is perfect agreement unless something was actually observed.
        return 1.0 if observed == 0.0 else 0.0

    return 1.0 - (observed / expected)


def _validated_confusion(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    categories: Sequence[str],
    /,
) -> list[list[int]]:
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Rater sequences differ in length: {len(rater_a)} vs {len(rater_b)}")

    if not rater_a:
        raise ValueError("Cannot compute kappa over empty sequences")

    if len(set(categories)) != len(categories):
        raise ValueError(f"Categories contain duplicate values: {categories}")

    return _confusion(rater_a, rater_b, categories)


def cohen_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    /,
    *,
    categories: Sequence[str] = EVALUATION_SCORE_LEVELS,
) -> float:
    """
    Compute unweighted (nominal) Cohen's kappa between two raters.

    Parameters
    ----------
    rater_a : Sequence[str]
        Category labels assigned by the first rater.
    rater_b : Sequence[str]
        Category labels assigned by the second rater (aligned with ``rater_a``).
    categories : Sequence[str]
        Full set of possible category labels, by default ``EVALUATION_SCORE_LEVELS``.

    Returns
    -------
    float
        Cohen's kappa in [-1, 1]; 1.0 is perfect agreement, 0.0 is chance-level.

    Raises
    ------
    ValueError
        If sequences are empty, differ in length, or contain unknown categories.
    """
    return _weighted_kappa(
        _validated_confusion(rater_a, rater_b, categories),
        lambda i, j: 0.0 if i == j else 1.0,
    )


def quadratic_weighted_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    /,
    *,
    categories: Sequence[str] = EVALUATION_SCORE_LEVELS,
) -> float:
    """
    Compute quadratic-weighted Cohen's kappa between two raters.

    The quadratic weighting penalizes near-miss disagreements less than distant
    ones, making it the preferred variant for ordered rating scales.

    Parameters
    ----------
    rater_a : Sequence[str]
        Category labels assigned by the first rater.
    rater_b : Sequence[str]
        Category labels assigned by the second rater (aligned with ``rater_a``).
    categories : Sequence[str]
        Full set of possible category labels in order, by default ``EVALUATION_SCORE_LEVELS``.

    Returns
    -------
    float
        Quadratic-weighted kappa in [-1, 1].

    Raises
    ------
    ValueError
        If sequences are empty, differ in length, or contain unknown categories.
    """
    matrix: list[list[int]] = _validated_confusion(rater_a, rater_b, categories)
    # Largest possible squared distance; falls back to 1.0 for a single-category
    # scale so the weight stays well-defined (every weight is then 0 anyway).
    max_distance: float = float((len(matrix) - 1) ** 2) or 1.0
    return _weighted_kappa(
        matrix,
        lambda i, j: ((i - j) ** 2) / max_distance,
    )
