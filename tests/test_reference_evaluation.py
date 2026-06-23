import pytest
from haiway import State

from draive.evaluation import (
    EvaluationReference,
    EvaluationScoreValue,
    Evaluator,
    evaluator,
    reference_conformance,
)

# --- EvaluationReference construction -------------------------------------------------


def test_reference_of_single_point_has_zero_width_window() -> None:
    reference = EvaluationReference.of("good")
    assert reference.lower == pytest.approx(0.6)
    assert reference.upper == pytest.approx(0.6)


def test_reference_of_applies_symmetric_percentage_tolerance() -> None:
    # tolerance is a fraction of the target: 0.6 * 0.2 = 0.12 margin
    reference = EvaluationReference.of("good", tolerance=0.2)
    assert reference.lower == pytest.approx(0.48)
    assert reference.upper == pytest.approx(0.72)


def test_reference_of_tolerance_scales_with_target() -> None:
    # the same percentage yields a wider margin for a larger target
    low = EvaluationReference.of("fair", tolerance=0.5)  # 0.4 * 0.5 = 0.2
    assert low.lower == pytest.approx(0.2)
    assert low.upper == pytest.approx(0.6)


def test_reference_of_clamps_window_to_unit_range() -> None:
    reference = EvaluationReference.of("perfect", tolerance=0.3)  # 1.0 * 0.3 = 0.3
    assert reference.lower == pytest.approx(0.7)
    assert reference.upper == pytest.approx(1.0)  # clamped, not 1.3


def test_reference_of_rejects_negative_tolerance() -> None:
    with pytest.raises(AssertionError):
        EvaluationReference.of("good", tolerance=-0.1)


def test_reference_between_accepts_descending_named_levels() -> None:
    reference = EvaluationReference.between("perfect", "good")
    assert reference.lower == pytest.approx(1.0)
    assert reference.upper == pytest.approx(0.6)


def test_reference_between_rejects_ascending_bounds() -> None:
    with pytest.raises(AssertionError):
        EvaluationReference.between("good", "perfect")


def test_reference_contains_is_inclusive() -> None:
    reference = EvaluationReference(lower=0.4, upper=0.6)
    assert reference.contains(0.4)
    assert reference.contains(0.5)
    assert reference.contains(0.6)
    assert not reference.contains(0.39)
    assert not reference.contains(0.61)


# --- reference_conformance ------------------------------------------------------------


def test_conformance_inside_window_is_perfect() -> None:
    reference = EvaluationReference(lower=0.4, upper=0.6)
    assert reference_conformance(0.5, reference) == 1.0


def test_conformance_nominal_rejects_any_miss() -> None:
    reference = EvaluationReference.of("good")
    assert reference_conformance(0.4, reference, weighting="nominal") == 0.0
    assert reference_conformance(0.6, reference, weighting="nominal") == 1.0


def test_conformance_quadratic_decays_with_squared_gap() -> None:
    reference = EvaluationReference.of("good")  # window [0.6, 0.6]
    # gap of 0.2 -> 1 - 0.2**2 = 0.96
    assert reference_conformance(0.4, reference) == pytest.approx(0.96)
    # gap measured from the nearest boundary of a wider window
    wide = EvaluationReference(lower=0.4, upper=0.6)
    assert reference_conformance(0.8, wide) == pytest.approx(1.0 - 0.2**2)


def test_conformance_quadratic_floors_at_zero() -> None:
    reference = EvaluationReference.of("none")  # window [0.0, 0.0]
    assert reference_conformance(1.0, reference) == 0.0  # 1 - 1**2


# --- Evaluator.referenced -------------------------------------------------------------


class _Case(State):
    predicted: EvaluationScoreValue
    expected: EvaluationReference


@evaluator(name="prediction")
async def _prediction(case: _Case) -> EvaluationScoreValue:
    return case.predicted


@pytest.mark.asyncio
async def test_referenced_replaces_score_with_conformance() -> None:
    referenced = Evaluator.referenced(
        _prediction.prepared(),
        reference=lambda case: case.expected,
    )

    result = await referenced(
        _Case(predicted="good", expected=EvaluationReference(lower=0.6, upper=1.0)),
    )

    assert result.score == 1.0
    assert result.passed
    assert result.evaluator == "prediction"
    assert result.meta["predicted_score"] == pytest.approx(0.6)
    assert result.meta["predicted_level"] == "good"
    assert result.meta["reference_lower"] == pytest.approx(0.6)
    assert result.meta["reference_upper"] == pytest.approx(1.0)
    assert result.meta["within_reference"] is True
    assert result.meta["reference_weighting"] == "quadratic"


@pytest.mark.asyncio
async def test_referenced_scores_miss_with_quadratic_falloff() -> None:
    referenced = Evaluator.referenced(
        _prediction.prepared(),
        reference=lambda case: case.expected,
    )

    result = await referenced(
        _Case(predicted="fair", expected=EvaluationReference.of("good")),
    )

    # predicted 0.4 vs window [0.6, 0.6]: gap 0.2 -> 0.96
    assert result.score == pytest.approx(0.96)
    assert result.meta["within_reference"] is False


@pytest.mark.asyncio
async def test_referenced_preserves_threshold() -> None:
    referenced = Evaluator.referenced(
        _prediction.with_threshold("excellent").prepared(),
        reference=lambda case: case.expected,
        weighting="nominal",
    )

    result = await referenced(
        _Case(predicted="poor", expected=EvaluationReference.of("good")),
    )

    assert result.threshold == pytest.approx(0.8)
    assert result.score == 0.0  # nominal miss
    assert not result.passed


@pytest.mark.asyncio
async def test_referenced_resolves_via_attribute_path() -> None:
    referenced = Evaluator.referenced(
        _prediction.prepared(),
        reference=_Case._.expected,
    )

    result = await referenced(
        _Case(predicted="excellent", expected=EvaluationReference(lower=0.6, upper=1.0)),
    )

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_referenced_constant_window_applies_to_every_value() -> None:
    referenced = Evaluator.referenced(
        _prediction.prepared(),
        reference=EvaluationReference(lower=0.6, upper=1.0),
    )

    inside = await referenced(
        _Case(predicted="perfect", expected=EvaluationReference.of("none")),
    )
    outside = await referenced(
        _Case(predicted="none", expected=EvaluationReference.of("perfect")),
    )

    assert inside.score == 1.0
    assert outside.score == pytest.approx(0.64)  # 0.0 vs [0.6, 1.0]: gap 0.6 -> 1 - 0.36 = 0.64


@pytest.mark.asyncio
async def test_referenced_treats_raw_value_as_exact_point() -> None:
    class _RawCase(State):
        predicted: EvaluationScoreValue
        expected: EvaluationScoreValue

    @evaluator(name="prediction")
    async def predict(case: _RawCase) -> EvaluationScoreValue:
        return case.predicted

    referenced = Evaluator.referenced(
        predict.prepared(),
        reference=lambda case: case.expected,
        weighting="nominal",
    )

    match = await referenced(_RawCase(predicted="good", expected="good"))
    miss = await referenced(_RawCase(predicted="good", expected="fair"))

    assert match.score == 1.0
    assert miss.score == 0.0
