from asyncio import gather
from collections.abc import Callable, Sequence
from typing import Protocol, overload, runtime_checkable

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.parameters import DataModel, Field
from draive.types import frozenlist
from draive.utils import freeze

__all__ = [
    "evaluation_scenario",
    "ScenarioEvaluator",
    "ScenarioEvaluatorDefinition",
    "ScenarioEvaluatorResult",
]


class ScenarioEvaluatorResult(DataModel):
    name: str = Field(
        description="Name of evaluated scenario",
    )
    evaluations: frozenlist[EvaluatorResult] = Field(
        description="Scenario evaluation results",
    )

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.evaluations)


@runtime_checkable
class PreparedScenarioEvaluator[Value](Protocol):
    async def __call__(
        self,
        value: Value,
    ) -> ScenarioEvaluatorResult: ...


@runtime_checkable
class ScenarioEvaluatorDefinition[Value, **Args](Protocol):
    @property
    def __name__(self) -> str: ...

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Sequence[PreparedEvaluator[Value]] | PreparedEvaluator[Value]: ...


class ScenarioEvaluator[Value, **Args]:
    def __init__(
        self,
        name: str,
        definition: ScenarioEvaluatorDefinition[Value, Args],
    ) -> None:
        self.name: str = name
        self._definition: ScenarioEvaluatorDefinition[Value, Args] = definition

        freeze(self)

    def prepared(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> PreparedScenarioEvaluator[Value]:
        prepared_evaluators: Sequence[PreparedEvaluator[Value]]
        match self._definition(*args, **kwargs):
            case [*evaluators]:
                prepared_evaluators = evaluators

            case evaluator:
                prepared_evaluators = (evaluator,)

        async def evaluate(
            value: Value,
        ) -> ScenarioEvaluatorResult:
            return ScenarioEvaluatorResult(
                name=self.name,
                evaluations=tuple(
                    await gather(
                        *[evaluator(value) for evaluator in prepared_evaluators],
                        return_exceptions=False,
                    ),
                ),
            )

        return evaluate

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> ScenarioEvaluatorResult:
        prepared_evaluators: Sequence[PreparedEvaluator[Value]]
        match self._definition(*args, **kwargs):
            case [*evaluators]:
                prepared_evaluators = evaluators

            case evaluator:
                prepared_evaluators = (evaluator,)

        return ScenarioEvaluatorResult(
            name=self.name,
            evaluations=tuple(
                await gather(
                    *[evaluator(value) for evaluator in prepared_evaluators],
                    return_exceptions=False,
                ),
            ),
        )


@overload
def evaluation_scenario[Value, **Args](
    definition: ScenarioEvaluatorDefinition[Value, Args],
    /,
) -> ScenarioEvaluator[Value, Args]: ...


@overload
def evaluation_scenario[Value, **Args](
    *,
    name: str,
) -> Callable[
    [ScenarioEvaluatorDefinition[Value, Args]],
    ScenarioEvaluator[Value, Args],
]: ...


def evaluation_scenario[Value, **Args](
    definition: ScenarioEvaluatorDefinition[Value, Args] | None = None,
    *,
    name: str | None = None,
) -> (
    Callable[
        [ScenarioEvaluatorDefinition[Value, Args]],
        ScenarioEvaluator[Value, Args],
    ]
    | ScenarioEvaluator[Value, Args]
):
    def wrap(
        definition: ScenarioEvaluatorDefinition[Value, Args],
    ) -> ScenarioEvaluator[Value, Args]:
        return ScenarioEvaluator(
            name=name or definition.__name__,
            definition=definition,
        )

    if definition:
        return wrap(definition)

    else:
        return wrap
