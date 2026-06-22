import pytest

from draive.evaluation import (
    EvaluationScore,
    EvaluationScoreValue,
    EvaluatorResult,
    cohen_kappa,
    quadratic_weighted_kappa,
)
from draive.evaluators import cohen_kappa_evaluator


def test_cohen_kappa_textbook_example() -> None:
    # Classic 2x2 agreement example: 50 items, confusion matrix
    #   [[20, 5], [10, 15]] -> observed=0.70, expected=0.50 -> kappa=0.40.
    # For a binary scale quadratic-weighted kappa coincides with the nominal one.
    rater_a: list[str] = ["perfect"] * 25 + ["none"] * 25
    rater_b: list[str] = ["perfect"] * 20 + ["none"] * 5 + ["perfect"] * 10 + ["none"] * 15
    categories = ("none", "perfect")

    assert cohen_kappa(rater_a, rater_b, categories=categories) == pytest.approx(0.4)
    assert quadratic_weighted_kappa(rater_a, rater_b, categories=categories) == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_perfect_agreement() -> None:
    ratings: list[EvaluationScoreValue] = ["poor", "good", "excellent", "perfect", "fair"]

    result = await cohen_kappa_evaluator(ratings, reference=ratings)

    assert result.score == 1.0
    assert result.passed
    assert result.meta["exact_agreement"] == 1.0
    assert result.meta["cohen_kappa"] == 1.0
    assert result.meta["quadratic_weighted_kappa"] == 1.0
    assert result.meta["sample_count"] == 5


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_disagreement_is_clamped() -> None:
    # Systematic inversion produces negative agreement which must clamp to 0.0,
    # while the raw (negative) value is preserved in metadata.
    evaluated: list[EvaluationScoreValue] = ["none", "none", "perfect", "perfect"]
    reference: list[EvaluationScoreValue] = ["perfect", "perfect", "none", "none"]

    result = await cohen_kappa_evaluator(evaluated, reference=reference)

    assert result.score == 0.0
    assert not result.passed
    # default weighting is quadratic, so that variant drives the clamped score
    quadratic = result.meta["quadratic_weighted_kappa"]
    assert isinstance(quadratic, float)
    assert quadratic < 0.0


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_quadratic_is_default() -> None:
    evaluated: list[EvaluationScoreValue] = ["poor", "good", "good", "excellent", "perfect", "fair"]
    reference: list[EvaluationScoreValue] = [
        "fair",
        "good",
        "excellent",
        "excellent",
        "perfect",
        "poor",
    ]

    result = await cohen_kappa_evaluator(evaluated, reference=reference)

    quadratic = result.meta["quadratic_weighted_kappa"]
    assert isinstance(quadratic, float)
    assert result.score == pytest.approx(max(0.0, quadratic))
    assert result.meta["weighting"] == "quadratic"


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_nominal_weighting_selects_nominal() -> None:
    evaluated: list[EvaluationScoreValue] = ["poor", "good", "excellent", "perfect", "fair", "good"]
    reference: list[EvaluationScoreValue] = ["poor", "fair", "good", "perfect", "fair", "excellent"]

    result = await cohen_kappa_evaluator(evaluated, reference=reference, weighting="nominal")

    nominal = result.meta["cohen_kappa"]
    assert isinstance(nominal, float)
    assert result.score == pytest.approx(max(0.0, nominal))
    assert result.meta["weighting"] == "nominal"


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_accepts_mixed_rating_types() -> None:
    evaluated: list[EvaluatorResult | EvaluationScore | EvaluationScoreValue] = [
        EvaluatorResult.of("e", score=0.6, threshold=0.5),  # -> "good"
        EvaluationScore.of(0.8),  # -> "excellent"
        "perfect",
    ]
    reference: list[EvaluationScoreValue] = ["good", "excellent", "perfect"]

    result = await cohen_kappa_evaluator(evaluated, reference=reference)

    assert result.score == 1.0
    assert result.meta["exact_agreement"] == 1.0


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_empty_ratings() -> None:
    result = await cohen_kappa_evaluator([], reference=[])

    assert result.score == 0.0
    assert result.meta["comment"] == "Ratings were empty!"


@pytest.mark.asyncio
async def test_cohen_kappa_evaluator_length_mismatch_fails() -> None:
    # Misaligned inputs yield a failed result carrying an explanatory comment.
    evaluated: list[EvaluationScoreValue] = ["good", "poor"]
    reference: list[EvaluationScoreValue] = ["good"]

    result = await cohen_kappa_evaluator(evaluated, reference=reference)

    assert result.score == 0.0
    assert not result.passed
    comment = result.meta["comment"]
    assert isinstance(comment, str)
    assert "2 vs 1" in comment
    assert result.meta["sample_count"] == 0
