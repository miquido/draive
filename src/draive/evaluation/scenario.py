from asyncio import gather
from collections.abc import Callable, Iterable
from typing import Protocol, overload, runtime_checkable

from draive.evaluation.evaluator import CaseEvaluationResult, PreparedCaseEvaluator
from draive.parameters import DataModel, Field
from draive.types import frozenlist
from draive.utils import freeze

__all__ = [
    "evaluation_scenario",
    "ScenarioEvaluationResult",
    "ScenarioEvaluator",
    "ScenarioDefinition",
]


class ScenarioEvaluationResult(DataModel):
    name: str = Field(
        description="Name of evaluated scenario",
    )
    cases: frozenlist[CaseEvaluationResult] = Field(
        description="List of evaluated scenario cases",
    )

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)


@runtime_checkable
class PreparedScenarioEvaluator[Value](Protocol):
    async def __call__(
        self,
        value: Value,
    ) -> ScenarioEvaluationResult: ...


@runtime_checkable
class ScenarioDefinition[Value, **Args](Protocol):
    @property
    def __name__(self) -> str: ...

    def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Iterable[PreparedCaseEvaluator[Value]]: ...


class ScenarioEvaluator[Value, **Args]:
    def __init__(
        self,
        name: str,
        definition: ScenarioDefinition[Value, Args],
    ) -> None:
        self.name: str = name
        self._definition: ScenarioDefinition[Value, Args] = definition

        freeze(self)

    def prepared(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> PreparedScenarioEvaluator[Value]:
        prepared_cases: Iterable[PreparedCaseEvaluator[Value]] = self._definition(*args, **kwargs)

        async def evaluate(
            value: Value,
        ) -> ScenarioEvaluationResult:
            return ScenarioEvaluationResult(
                name=self.name,
                cases=tuple(
                    await gather(
                        *[case(value=value) for case in prepared_cases],
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
    ) -> ScenarioEvaluationResult:
        return ScenarioEvaluationResult(
            name=self.name,
            cases=tuple(
                await gather(
                    *[case(value=value) for case in self._definition(*args, **kwargs)],
                    return_exceptions=False,
                ),
            ),
        )


@overload
def evaluation_scenario[Value, **Args](
    definition: ScenarioDefinition[Value, Args],
    /,
) -> ScenarioEvaluator[Value, Args]: ...


@overload
def evaluation_scenario[Value, **Args](
    *,
    name: str,
) -> Callable[
    [ScenarioDefinition[Value, Args]],
    ScenarioEvaluator[Value, Args],
]: ...


def evaluation_scenario[Value, **Args](
    definition: ScenarioDefinition[Value, Args] | None = None,
    *,
    name: str | None = None,
) -> (
    Callable[
        [ScenarioDefinition[Value, Args]],
        ScenarioEvaluator[Value, Args],
    ]
    | ScenarioEvaluator[Value, Args]
):
    def wrap(
        definition: ScenarioDefinition[Value, Args],
    ) -> ScenarioEvaluator[Value, Args]:
        return ScenarioEvaluator(
            name=name or definition.__name__,
            definition=definition,
        )

    if definition:
        return wrap(definition)

    else:
        return wrap
