from asyncio import gather
from collections.abc import Callable, Sequence
from typing import Protocol, Self, cast, overload, runtime_checkable

from haiway import AttributePath, ScopeContext, ctx, freeze

from draive.commons import META_EMPTY, Meta
from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.parameters import DataModel, Field

__all__ = (
    "EvaluationScenarioResult",
    "ScenarioEvaluator",
    "ScenarioEvaluatorDefinition",
    "ScenarioEvaluatorResult",
    "evaluation_scenario",
)


class ScenarioEvaluatorResult(DataModel):
    scenario: str = Field(
        description="Name of evaluated scenario",
    )
    evaluations: Sequence[EvaluatorResult] = Field(
        description="Scenario evaluation results",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )

    @property
    def passed(self) -> bool:
        # empty evaluations is equivalent of failure
        return len(self.evaluations) > 0 and all(case.passed for case in self.evaluations)

    def report(
        self,
        *,
        include_passed: bool = True,
        include_details: bool = True,
    ) -> str:
        report: str = "\n- ".join(
            result.report(include_details=include_details)
            for result in self.evaluations
            if include_passed or not result.passed
        )

        if report:  # nonempty report
            if include_details:
                meta_values: str = (
                    f"\n{'\n'.join(f'{key}: {value}' for key, value in self.meta.items())}"
                    if self.meta
                    else "N/A"
                )
                return f"Scenario {self.scenario}, meta: {meta_values}\n---\n{report}"

            else:
                return f"Scenario {self.scenario}:\n{report}"

        elif not self.evaluations:
            return f"Scenario {self.scenario} empty!"

        else:
            return f"Scenario {self.scenario} passed!"

    @property
    def relative_score(self) -> float:
        if not self.evaluations:
            return 0

        passed: int = 0
        for evaluation in self.evaluations:
            if evaluation.passed:
                passed += 1

        return passed / len(self.evaluations)


class EvaluationScenarioResult(DataModel):
    @classmethod
    async def evaluating[Value](
        cls,
        value: Value,
        /,
        evaluators: PreparedEvaluator[Value],
        *_evaluators: PreparedEvaluator[Value],
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            evaluations=tuple(
                await gather(
                    *[evaluator(value) for evaluator in [evaluators, *_evaluators]],
                    return_exceptions=False,
                ),
            ),
            meta=meta if meta is not None else META_EMPTY,
        )

    evaluations: Sequence[EvaluatorResult] = Field(
        description="Scenario evaluation results",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
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
        execution_context: ScopeContext | None,
        meta: Meta | None,
    ) -> None:
        assert "\n" not in name  # nosec: B101

        self.name: str = name.lower().replace(" ", "_")
        self._definition: ScenarioEvaluatorDefinition[Value, Args] = definition
        self._execution_context: ScopeContext | None = execution_context
        self.meta: Meta = meta if meta is not None else META_EMPTY

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

    def with_name(
        self,
        name: str,
        /,
    ) -> Self:
        return self.__class__(
            name=name,
            definition=self._definition,
            execution_context=self._execution_context,
            meta=self.meta,
        )

    def with_execution_context(
        self,
        context: ScopeContext,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            execution_context=context,
            meta=self.meta,
        )

    def with_meta(
        self,
        meta: Meta,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            execution_context=self._execution_context,
            meta={**self.meta, **meta} if self.meta else meta,
        )

    def contra_map[Mapped](
        self,
        mapping: Callable[[Mapped], Value] | AttributePath[Mapped, Value] | Value,
        /,
    ) -> "ScenarioEvaluator[Mapped, Args]":
        mapper: Callable[[Mapped], Value]
        match mapping:
            case Callable() as function:  # pyright: ignore[reportUnknownVariableType]
                mapper = function

            case path:
                assert isinstance(  # nosec: B101
                    path, AttributePath
                ), "Prepare parameter path by using Self._.path.to.property"
                mapper = cast(AttributePath[Mapped, Value], path).__call__

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
            execution_context=self._execution_context,
            meta=self.meta,
        )

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> ScenarioEvaluatorResult:
        result: ScenarioEvaluatorResult
        if context := self._execution_context:
            async with context:
                result = await self._evaluate(
                    value,
                    *args,
                    **kwargs,
                )

        else:
            async with ctx.scope(f"evaluation.scenario.{self.name}"):
                result = await self._evaluate(
                    value,
                    *args,
                    **kwargs,
                )

        ctx.record(
            metric=f"evaluation.scenario.{result.scenario}.score",
            value=result.relative_score,
            unit="%",
            attributes={
                "evaluation.scenario.evaluations": [
                    evaluation.evaluator for evaluation in result.evaluations
                ],
                "evaluation.passed": result.passed,
            },
        )
        return result

    async def _evaluate(
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
                    meta: Meta
                    if self.meta:
                        if result.meta:
                            meta = {**self.meta, **result.meta}

                        else:
                            meta = self.meta

                    else:
                        meta = result.meta

                    return ScenarioEvaluatorResult(
                        scenario=self.name,
                        evaluations=result.evaluations,
                        meta=meta,
                    )

                case [*results]:
                    return ScenarioEvaluatorResult(
                        scenario=self.name,
                        evaluations=tuple(results),
                        meta=self.meta,
                    )

        except Exception as exc:
            ctx.log_error(
                f"Scenario evaluator `{self.name}` failed, using empty fallback result",
                exception=exc,
            )

            return ScenarioEvaluatorResult(
                scenario=self.name,
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
    name: str | None = None,
    execution_context: ScopeContext | None = None,
    meta: Meta | None = None,
) -> Callable[
    [ScenarioEvaluatorDefinition[Value, Args]],
    ScenarioEvaluator[Value, Args],
]: ...


def evaluation_scenario[Value, **Args](  # pyright: ignore[reportInconsistentOverload] - this seems to be pyright false positive/error
    definition: ScenarioEvaluatorDefinition[Value, Args] | None = None,
    /,
    *,
    name: str | None = None,
    execution_context: ScopeContext | None = None,
    meta: Meta | None = None,
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
            execution_context=execution_context,
            meta=meta,
        )

    if definition:
        return wrap(definition)

    else:
        return wrap
