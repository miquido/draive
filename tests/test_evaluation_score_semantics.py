import pytest
from haiway import BasicValue, ctx

from draive.evaluation import (
    EvaluationScore,
    Evaluator,
    EvaluatorResult,
    PreparedEvaluator,
    evaluator,
)
from draive.evaluation.value import evaluation_score_level, evaluation_score_value


@pytest.mark.parametrize(
    ("score", "other", "less", "greater"),
    (
        (0.5, 0.75, True, False),
        (0.5, 0.25, False, True),
        (0.5, 0.5, False, False),
    ),
)
def test_evaluation_score_float_comparisons(
    score: float,
    other: float,
    less: bool,
    greater: bool,
) -> None:
    value = EvaluationScore.of(score)

    assert (value < other) is less
    assert (value > other) is greater
    assert (value <= other) is (less or score == other)
    assert (value >= other) is (greater or score == other)


def test_evaluation_score_perfect_is_at_least_integer_one() -> None:
    assert EvaluationScore.of(1.0) >= 1
    assert EvaluationScore.of(0.0) <= 0
    assert EvaluationScore.of(0.5) > 0
    assert EvaluationScore.of(0.5) < 1


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0.0, "none"),
        (0.19999, "none"),
        (0.2, "poor"),
        (0.4, "fair"),
        (0.6, "good"),
        (0.8, "excellent"),
        (0.998, "excellent"),
        (0.999, "perfect"),
        (1.0, "perfect"),
        (True, "perfect"),
        (False, "none"),
        (0, "none"),
        (1, "perfect"),
    ),
)
def test_evaluation_score_level_boundaries(value: float | bool, expected: str) -> None:
    assert evaluation_score_level(value) == expected


@pytest.mark.parametrize(
    "value",
    (-0.0001, 1.0001, float("nan"), float("inf"), 2, -1, "GOOD", "unknown", None),
)
def test_evaluation_score_value_rejects_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        evaluation_score_value(value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("score", "threshold", "passed", "performance"),
    (
        (0.5, 0.5, True, 100.0),
        (0.49, 0.5, False, 98.0),
        (1.0, 0.5, True, 200.0),
        (0.0, 0.0, True, 100.0),
        (0.3, 0.0, True, 100.0),
    ),
)
def test_evaluator_result_threshold_boundaries(
    score: float,
    threshold: float,
    passed: bool,
    performance: float,
) -> None:
    result = EvaluatorResult.of("check", score=score, threshold=threshold)

    assert result.passed is passed
    assert result.performance == pytest.approx(performance)


def test_evaluator_result_serialization_round_trip() -> None:
    result = EvaluatorResult.of(
        "check",
        score=EvaluationScore.of("good", meta={"comment": "fine"}),
        threshold="fair",
        meta={"extra": 1},
    )

    restored = EvaluatorResult.from_json(result.to_json())

    assert restored == result
    assert restored.meta["comment"] == "fine"
    assert restored.meta["extra"] == 1


@evaluator(name="constant_score", threshold=0.35)
async def _constant_score(value: float) -> float:
    return value


@pytest.mark.asyncio
async def test_evaluator_average_of_equal_scores_meets_equal_threshold() -> None:
    async with ctx.scope("test"):
        result = await Evaluator.average(
            _constant_score.prepared(),
            _constant_score.prepared(),
            _constant_score.prepared(),
            threshold=0.35,
        )(0.35)

    assert result.passed


@pytest.mark.asyncio
async def test_evaluator_failure_is_isolated_into_zero_score() -> None:
    @evaluator(name="broken", threshold=0.5)
    async def broken(value: float) -> float:
        raise RuntimeError("judge down")

    async with ctx.scope("test"):
        result = await broken(0.5)

    assert result.score == 0.0
    assert not result.passed
    assert "judge down" in str(result.meta["error"])


@pytest.mark.asyncio
async def test_evaluator_out_of_range_result_is_isolated() -> None:
    @evaluator(name="overshoot", threshold=0.5)
    async def overshoot(value: float) -> float:
        return 1.5

    async with ctx.scope("test"):
        result = await overshoot(0.5)

    assert result.score == 0.0
    assert "error" in result.meta


@pytest.mark.parametrize("score", (0.0, 0.5, 1.0))
def test_zero_threshold_always_passes_and_reports_full_performance(score: float) -> None:
    result = EvaluatorResult.of("any", score=score, threshold=0.0)

    assert result.passed
    assert result.performance == 100.0
    assert "(100.00%)" in result.report(detailed=False)


def test_evaluator_result_ordering_with_foreign_types_returns_not_implemented() -> None:
    result = EvaluatorResult.of("any", score=0.5, threshold=0.5)

    for operation in (
        lambda: result < 1,  # pyright: ignore[reportOperatorIssue]
        lambda: result <= "x",  # pyright: ignore[reportOperatorIssue]
        lambda: result > None,  # pyright: ignore[reportOperatorIssue]
        lambda: result >= 0.5,  # pyright: ignore[reportOperatorIssue]
    ):
        with pytest.raises(TypeError):
            operation()

    assert result.__lt__(1) is NotImplemented  # pyright: ignore[reportArgumentType]
    assert result != 1


def test_evaluator_result_ordering_of_different_evaluators_still_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _ = EvaluatorResult.of("a", score=0.5, threshold=0.5) < EvaluatorResult.of(
            "b", score=0.5, threshold=0.5
        )


@pytest.mark.asyncio
async def test_evaluator_average_keeps_error_of_failed_evaluation() -> None:
    @evaluator(name="broken", threshold=0.5)
    async def broken(value: float) -> float:
        raise RuntimeError("judge down")

    @evaluator(name="fine", threshold=0.5)
    async def fine(value: float) -> float:
        return 1.0

    async with ctx.scope("test"):
        result = await Evaluator.average(
            broken.prepared(),
            fine.prepared(),
            threshold=0.5,
        )(0.5)

    assert result.score == pytest.approx(0.5)
    assert "judge down" in str(result.meta["error"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (42, True, ["a"], {"k": "v"}))
async def test_evaluator_average_ignores_non_text_error_meta(error: BasicValue) -> None:
    async def custom(value: float, /) -> EvaluatorResult:
        return EvaluatorResult.of(
            "custom",
            score=0.5,
            threshold=0.5,
            meta={"error": error},
        )

    prepared: PreparedEvaluator[float] = custom
    async with ctx.scope("test"):
        result = await Evaluator.average(prepared, prepared, threshold=0.5)(0.5)

    assert result.score == pytest.approx(0.5)
    assert "error" not in result.meta
