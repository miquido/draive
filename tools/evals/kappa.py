from collections.abc import Sequence
from typing import Final

from haiway import State

__all__ = (
    "BIN_LABELS",
    "BIN_VALUES",
    "KappaReport",
    "cohen_kappa",
    "kappa_report",
    "quadratic_weighted_kappa",
    "quantize_score",
)


BIN_LABELS: Final[tuple[str, ...]] = (
    "none",
    "poor",
    "fair",
    "good",
    "excellent",
    "perfect",
)

# Midpoint values from draive.evaluation.value
BIN_VALUES: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def quantize_score(value: float, /) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Score outside [0, 1]: {value}")

    closest_index: int = min(
        range(len(BIN_VALUES)),
        key=lambda index: abs(value - BIN_VALUES[index]),
    )
    return BIN_LABELS[closest_index]


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


def _kappa_from_matrix(matrix: list[list[int]], /) -> float:
    size: int = len(matrix)
    total: int = sum(sum(row) for row in matrix)

    observed_agreement: float = sum(matrix[i][i] for i in range(size)) / total

    expected_agreement: float = 0.0
    for i in range(size):
        row_total: int = sum(matrix[i])
        col_total: int = sum(matrix[r][i] for r in range(size))
        expected_agreement += (row_total * col_total) / (total * total)

    denominator: float = 1.0 - expected_agreement
    if denominator == 0.0:
        # All raters always picked the same category - perfect by construction.
        return 1.0 if observed_agreement == 1.0 else 0.0

    return (observed_agreement - expected_agreement) / denominator


def _quadratic_weighted_kappa_from_matrix(matrix: list[list[int]], /) -> float:
    size: int = len(matrix)
    total: int = sum(sum(row) for row in matrix)

    row_totals: list[int] = [sum(matrix[i]) for i in range(size)]
    col_totals: list[int] = [sum(matrix[r][i] for r in range(size)) for i in range(size)]

    denominator_max: float = float((size - 1) ** 2)
    observed_numer: float = 0.0
    expected_numer: float = 0.0
    for i in range(size):
        for j in range(size):
            weight: float = ((i - j) ** 2) / denominator_max
            observed_numer += weight * matrix[i][j]
            expected_numer += weight * (row_totals[i] * col_totals[j]) / total

    if expected_numer == 0.0:
        return 1.0 if observed_numer == 0.0 else 0.0

    return 1.0 - (observed_numer / expected_numer)


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

    return _confusion(rater_a, rater_b, categories)


def cohen_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    /,
    *,
    categories: Sequence[str] = BIN_LABELS,
) -> float:
    return _kappa_from_matrix(_validated_confusion(rater_a, rater_b, categories))


def quadratic_weighted_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    /,
    *,
    categories: Sequence[str] = BIN_LABELS,
) -> float:
    return _quadratic_weighted_kappa_from_matrix(_validated_confusion(rater_a, rater_b, categories))


class KappaReport(State):
    sample_count: int
    human_bins: Sequence[str]
    evaluator_bins: Sequence[str]
    cohen_kappa: float
    quadratic_weighted_kappa: float
    exact_agreement: float

    def render(self) -> str:
        return "\n".join(
            (
                f"samples: {self.sample_count}",
                f"exact agreement: {self.exact_agreement * 100:.1f}%",
                f"Cohen's kappa (nominal):           {self.cohen_kappa:.3f}",
                f"Cohen's kappa (quadratic weighted): {self.quadratic_weighted_kappa:.3f}",
                "",
                "interpretation (Landis & Koch, 1977):",
                "  < 0.00 poor | 0.00-0.20 slight | 0.21-0.40 fair",
                "  0.41-0.60 moderate | 0.61-0.80 substantial | 0.81-1.00 almost perfect",
            )
        )


def kappa_report(
    human_scores: Sequence[float],
    evaluator_scores: Sequence[float],
    /,
) -> KappaReport:
    if len(human_scores) != len(evaluator_scores):
        raise ValueError(
            f"Score sequences differ in length: {len(human_scores)} vs {len(evaluator_scores)}"
        )

    if not human_scores:
        raise ValueError("Cannot compute kappa over empty sequences")

    human_bins: tuple[str, ...] = tuple(quantize_score(value) for value in human_scores)
    evaluator_bins: tuple[str, ...] = tuple(quantize_score(value) for value in evaluator_scores)

    exact_matches: int = sum(
        1 for human, evaluator in zip(human_bins, evaluator_bins, strict=True) if human == evaluator
    )

    matrix: list[list[int]] = _validated_confusion(human_bins, evaluator_bins, BIN_LABELS)

    return KappaReport(
        sample_count=len(human_bins),
        human_bins=human_bins,
        evaluator_bins=evaluator_bins,
        cohen_kappa=_kappa_from_matrix(matrix),
        quadratic_weighted_kappa=_quadratic_weighted_kappa_from_matrix(matrix),
        exact_agreement=exact_matches / len(human_bins),
    )
