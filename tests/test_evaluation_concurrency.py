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
