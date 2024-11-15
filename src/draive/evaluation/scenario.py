from asyncio import gather
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Self, cast, overload, runtime_checkable

from haiway import ctx, freeze, frozenlist

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.parameters import DataModel, Field
from draive.parameters.path import ParameterPath

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
    meta: Mapping[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
    )

    @property
    def passed(self) -> bool:
        # empty evaluations is equivalent of failure
        return len(self.evaluations) > 0 and all(case.passed for case in self.evaluations)

    def report(self) -> str:
        report: str = "\n- ".join(
            result.report() for result in self.evaluations if not result.passed
        )

        if report:  # nonempty report contains failing reports
            meta_values: str = (
                f"\n{'\n'.join(f'{key}: {value}' for key, value in self.meta.items())}"
                if self.meta
                else "N/A"
            )
            return f"Scenario {self.name}, meta: {meta_values}\n---\n{report}"

        elif not self.evaluations:
            return f"Scenario {self.name} empty!"

        else:
            return f"Scenario {self.name} passed!"


class EvaluationScenarioResult(DataModel):
    @classmethod
    async def evaluating[Value](
        cls,
        value: Value,
        /,
        evaluators: PreparedEvaluator[Value],
        *_evaluators: PreparedEvaluator[Value],
        meta: Mapping[str, str | float | int | bool | None] | None = None,
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
    meta: Mapping[str, str | float | int | bool | None] | None = Field(
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
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> None:
        self.name: str = name
        self._definition: ScenarioEvaluatorDefinition[Value, Args] = definition
        self.meta: Mapping[str, str | float | int | bool | None] | None = meta

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

    def with_meta(
        self,
        meta: Mapping[str, str | float | int | bool | None],
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            meta={**self.meta, **meta} if self.meta else meta,
        )

    def contra_map[Mapped](
        self,
        mapping: Callable[[Mapped], Value] | ParameterPath[Mapped, Value] | Value,
        /,
    ) -> "ScenarioEvaluator[Mapped, Args]":
        mapper: Callable[[Mapped], Value]
        match mapping:
            case Callable() as function:  # pyright: ignore[reportUnknownVariableType]
                mapper = function

            case path:
                assert isinstance(  # nosec: B101
                    path, ParameterPath
                ), "Prepare parameter path by using Self._.path.to.property"
                mapper = cast(ParameterPath[Mapped, Value], path).__call__

        async def evaluation(
            value: Mapped,
            /,
            *args: Args.args,
            **kwargs: Args.kwargs,
        ) -> Sequence[EvaluatorResult] | EvaluationScenarioResult:
            return await self._definition(
                mapper(value),
                *args,
                **kwargs,
            )

        return ScenarioEvaluator[Mapped, Args](
            name=self.name,
            definition=evaluation,
        )

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
                    meta: Mapping[str, str | float | int | bool | None] | None
                    if self.meta:
                        if result.meta:
                            meta = {**self.meta, **result.meta}

                        else:
                            meta = self.meta

                    elif result.meta:
                        meta = result.meta

                    else:
                        meta = None

                    return ScenarioEvaluatorResult(
                        name=self.name,
                        evaluations=result.evaluations,
                        meta=meta,
                    )

                case [*results]:
                    return ScenarioEvaluatorResult(
                        name=self.name,
                        evaluations=tuple(results),
                        meta=self.meta,
                    )

        except Exception as exc:
            ctx.log_error(
                f"Scenario evaluator `{self.name}` failed, using empty fallback result",
                exception=exc,
            )

            return ScenarioEvaluatorResult(
                name=self.name,
                evaluations=(),
                meta={**(self.meta or {}), "exception": str(exc)},
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


def evaluation_scenario[Value, **Args](  # pyright: ignore[reportInconsistentOverload] - this seems to be pyright false positive/error
    definition: ScenarioEvaluatorDefinition[Value, Args] | None = None,
    /,
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
