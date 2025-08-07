from collections.abc import Sequence
from typing import overload

from haiway import concurrently

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.evaluation.scenario import EvaluatorScenarioResult, PreparedEvaluatorScenario

__all__ = ("evaluate",)


@overload
async def evaluate[Value](
    value: Value,
    *evaluators: PreparedEvaluatorScenario[Value],
    concurrent_tasks: int = 2,
) -> Sequence[EvaluatorScenarioResult]: ...


@overload
async def evaluate[Value](
    value: Value,
    *evaluators: PreparedEvaluator[Value],
    concurrent_tasks: int = 2,
) -> Sequence[EvaluatorResult]: ...


@overload
async def evaluate[Value](
    value: Value,
    *evaluators: PreparedEvaluatorScenario[Value] | PreparedEvaluator[Value],
    concurrent_tasks: int = 2,
) -> Sequence[EvaluatorScenarioResult | EvaluatorResult]: ...


async def evaluate[Value](
    value: Value,
    *evaluators: PreparedEvaluatorScenario[Value] | PreparedEvaluator[Value],
    concurrent_tasks: int = 2,
) -> Sequence[EvaluatorScenarioResult | EvaluatorResult]:
    return await concurrently(
        (evaluator(value) for evaluator in evaluators),
        concurrent_tasks=concurrent_tasks,
        return_exceptions=False,
    )
