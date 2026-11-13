import pytest
from haiway import ctx

from draive.evaluation import Evaluator, evaluate, evaluator


@evaluator(name="constant", threshold="good")
async def constant_evaluator(value: float) -> float:
    return value


@pytest.mark.asyncio
async def test_evaluate_accepts_serial_concurrency() -> None:
    # a limit of one runs the evaluators one after another, which the underlying
    # concurrency helpers support - it used to be rejected as below the minimum
    async with ctx.scope("test"):
        results = await evaluate(
            1.0,
            constant_evaluator.prepared(),
            constant_evaluator.prepared(),
            concurrent_tasks=1,
        )

    assert [result.score for result in results] == [1.0, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_tasks", (1, 2))
async def test_evaluator_combinators_accept_serial_concurrency(concurrent_tasks: int) -> None:
    async with ctx.scope("test"):
        lowest = await Evaluator.lowest(
            constant_evaluator.prepared(),
            constant_evaluator.with_threshold("good").prepared(),
            concurrent_tasks=concurrent_tasks,
        )(0.2)
        highest = await Evaluator.highest(
            constant_evaluator.prepared(),
            constant_evaluator.prepared(),
            concurrent_tasks=concurrent_tasks,
        )(0.8)
        average = await Evaluator.average(
            constant_evaluator.prepared(),
            constant_evaluator.prepared(),
            threshold="good",
            concurrent_tasks=concurrent_tasks,
        )(0.5)

    assert lowest.score == pytest.approx(0.2)
    assert highest.score == pytest.approx(0.8)
    assert average.score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_evaluator_lowest_returns_real_result_when_all_exceed_threshold() -> None:
    async with ctx.scope("test"):
        lowest = await Evaluator.lowest(
            constant_evaluator.prepared(),
            constant_evaluator.prepared(),
        )(1.0)

    assert lowest.evaluator == "constant"
    assert lowest.score == pytest.approx(1.0)
    assert lowest.threshold == pytest.approx(0.6)


def test_evaluation_score_equality_and_hash_contract() -> None:
    from haiway import Meta

    from draive.evaluation import EvaluationScore

    first = EvaluationScore.of(0.5, meta=Meta.of({"a": "b"}))
    second = EvaluationScore.of(0.5)
    assert first == second
    assert hash(first) == hash(second)
    assert first == 0.5
    assert hash(first) == hash(0.5)
    assert first != None  # noqa: E711
    assert not first == "x"


@pytest.mark.asyncio
async def test_evaluator_result_equality_does_not_raise() -> None:
    async with ctx.scope("test"):
        result = await constant_evaluator.prepared()(0.5)
        other = await constant_evaluator.with_threshold("perfect").prepared()(0.5)

    assert result != None  # noqa: E711
    assert result != other
    assert result == result  # noqa: PLR0124


@evaluator(name="needs_reference", threshold="perfect")
async def _reference_evaluator(value: str, /, *, reference: str) -> float:
    return 1.0 if value == reference else 0.0


@pytest.mark.asyncio
async def test_evaluator_raises_for_invalid_arguments_instead_of_scoring() -> None:
    async with ctx.scope("test"):
        with pytest.raises(TypeError):
            await _reference_evaluator("text")  # pyright: ignore[reportCallIssue]

        result = await _reference_evaluator("text", reference="text")

    assert result.score == 1.0
