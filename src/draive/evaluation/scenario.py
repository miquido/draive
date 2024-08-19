from asyncio import gather
from collections.abc import Callable, Sequence
from typing import Protocol, Self, overload, runtime_checkable

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.parameters import DataModel, Field
from draive.scope import ctx
from draive.types import frozenlist
from draive.utils import freeze

__all__ = [
    "evaluation_scenario",
    "ScenarioEvaluator",
    "ScenarioEvaluatorDefinition",
    "ScenarioEvaluatorResult",
    "EvaluationScenarioResult",
]


class ScenarioEvaluatorResult(DataModel):
    name: str = Field(
        description="Name of evaluated scenario",
    )
    evaluations: frozenlist[EvaluatorResult] = Field(
        description="Scenario evaluation results",
    )
    meta: dict[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
    )

    @property
    def passed(self) -> bool:
        # empty evaluations is equivalent of failure
        return len(self.evaluations) > 0 and all(case.passed for case in self.evaluations)


class EvaluationScenarioResult(DataModel):
    @classmethod
    async def evaluating[Value](
        cls,
        value: Value,
        /,
        evaluators: PreparedEvaluator[Value],
        *_evaluators: PreparedEvaluator[Value],
        meta: dict[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            evaluations=tuple(
                await gather(
                    *[evaluator(value) for evaluator in [evaluators, *_evaluators]],
                    return_exceptions=False,
                ),
            ),
            meta=meta,
        )

    evaluations: frozenlist[EvaluatorResult] = Field(
        description="Scenario evaluation results",
    )
    meta: dict[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
    )


@runtime_checkable
class PreparedScenarioEvaluator[Value](Protocol):
    async def __call__(
        self,
        value: Value,
        /,
    ) -> ScenarioEvaluatorResult: ...


@runtime_checkable
class ScenarioEvaluatorDefinition[Value, **Args](Protocol):
    @property
    def __name__(self) -> str: ...

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Sequence[EvaluatorResult] | EvaluationScenarioResult: ...


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
        async def evaluate(
            value: Value,
        ) -> ScenarioEvaluatorResult:
            return await self(
                value,
                *args,
                **kwargs,
            )

        return evaluate

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> ScenarioEvaluatorResult:
        try:
            match await self._definition(
                value,
                *args,
                **kwargs,
            ):
                case EvaluationScenarioResult() as result:
                    return ScenarioEvaluatorResult(
                        name=self.name,
                        evaluations=result.evaluations,
                        meta=result.meta,
                    )

                case [*results]:
                    return ScenarioEvaluatorResult(
                        name=self.name,
                        evaluations=tuple(results),
                    )
        except Exception as exc:
            ctx.log_error(
                f"Scenario evaluator `{self.name}` failed, using empty fallback result",
                exception=exc,
            )

            return ScenarioEvaluatorResult(
                name=self.name,
                evaluations=(),
                meta={"exception": str(exc)},
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
