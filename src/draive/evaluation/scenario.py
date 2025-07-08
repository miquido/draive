from collections.abc import Callable, Sequence
from typing import Any, Protocol, Self, cast, overload, runtime_checkable

from haiway import AttributePath, ScopeContext, as_list, ctx, execute_concurrently

from draive.commons import META_EMPTY, Meta, MetaValues
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
        evaluations_report: str = "\n".join(
            result.report(include_details=include_details)
            for result in self.evaluations
            if include_passed or not result.passed
        )

        if evaluations_report:  # nonempty report
            if include_details:
                return (
                    f"<scenario name='{self.scenario}'>"
                    f"\n<relative_score>{self.relative_score * 100:.2f}%</relative_score>"
                    f"\n<evaluations>\n{evaluations_report}\n</evaluations>"
                    "\n</scenario>"
                )

            else:
                return f"Scenario {self.scenario}:\n{evaluations_report}"

        elif not self.evaluations:
            return f"Scenario {self.scenario} empty!"

        else:
            return f"Scenario {self.scenario} passed!"

    @property
    def relative_score(self) -> float:
        if not self.evaluations:
            return 0.0

        score: float = 0.0
        for evaluation in self.evaluations:
            score += evaluation.relative_score

        return score / len(self.evaluations)


class EvaluationScenarioResult(DataModel):
    @classmethod
    async def evaluating[Value](
        cls,
        value: Value,
        /,
        evaluators: PreparedEvaluator[Value],
        *_evaluators: PreparedEvaluator[Value],
        concurrent_tasks: int = 2,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        async def execute(
            evaluator: PreparedEvaluator[Value],
        ) -> EvaluatorResult:
            return await evaluator(value)

        return cls(
            evaluations=tuple(
                await execute_concurrently(
                    execute,
                    [evaluators, *_evaluators],
                    concurrent_tasks=concurrent_tasks,
                ),
            ),
            meta=Meta.of(meta),
        )

    @classmethod
    def merging(
        cls,
        result: Self,
        *results: Self,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        merged_evaluations: list[EvaluatorResult] = as_list(result.evaluations)
        merged_meta: Meta = result.meta
        for other in results:
            merged_evaluations.extend(other.evaluations)
            merged_meta = merged_meta.merged_with(other.meta)

        return cls(
            evaluations=merged_evaluations,
            meta=merged_meta.merged_with(Meta.of(meta)),
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
    __slots__ = (
        "_definition",
        "_execution_context",
        "meta",
        "name",
    )

    def __init__(
        self,
        name: str,
        definition: ScenarioEvaluatorDefinition[Value, Args],
        execution_context: ScopeContext | None,
        meta: Meta,
    ) -> None:
        assert "\n" not in name  # nosec: B101

        self.name: str
        object.__setattr__(
            self,
            "name",
            name.lower().replace(" ", "_"),
        )
        self._definition: ScenarioEvaluatorDefinition[Value, Args]
        object.__setattr__(
            self,
            "_definition",
            definition,
        )
        self._execution_context: ScopeContext | None
        object.__setattr__(
            self,
            "_execution_context",
            execution_context,
        )
        self.meta: Meta
        object.__setattr__(
            self,
            "meta",
            meta,
        )

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
        meta: Meta | MetaValues,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            execution_context=self._execution_context,
            meta=self.meta.merged_with(meta),
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
                            meta = self.meta.merged_with(result.meta)

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
                meta=self.meta.updated(exception=str(exc)),
            )

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> Any:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be modified"
        )

    def __delattr__(
        self,
        name: str,
    ) -> None:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be deleted"
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
    meta: Meta | MetaValues | None = None,
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
    meta: Meta | MetaValues | None = None,
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
            meta=Meta.of(meta),
        )

    if definition:
        return wrap(definition)

    else:
        return wrap
