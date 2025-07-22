from collections.abc import Callable, Collection, Sequence
from typing import Any, Protocol, Self, cast, overload, runtime_checkable

from haiway import (
    META_EMPTY,
    AttributePath,
    Meta,
    MetaValues,
    State,
    as_list,
    ctx,
    execute_concurrently,
)

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.parameters import DataModel, Field

__all__ = (
    "EvaluationScenarioResult",
    "EvaluatorScenario",
    "EvaluatorScenarioDefinition",
    "EvaluatorScenarioResult",
    "evaluator_scenario",
)


class EvaluationScenarioResult(DataModel):
    """
    Result of evaluating multiple evaluators on a value.

    Contains a collection of evaluation results from running multiple
    evaluators, along with metadata about the scenario evaluation.
    """

    @classmethod
    async def evaluating[Value](
        cls,
        value: Value,
        /,
        evaluator: PreparedEvaluator[Value],
        *evaluators: PreparedEvaluator[Value],
        concurrent_tasks: int = 2,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Create a result by evaluating multiple evaluators on a value.

        Parameters
        ----------
        value : Value
            The value to evaluate
        evaluator : PreparedEvaluator[Value]
            First evaluator to run
        *evaluators : PreparedEvaluator[Value]
            Additional evaluators to run
        concurrent_tasks : int, optional
            Maximum number of concurrent evaluation tasks, by default 2
        meta : Meta | MetaValues | None, optional
            Additional metadata for the result

        Returns
        -------
        Self
            New EvaluationScenarioResult with all evaluation results
        """

        async def execute(
            evaluator: PreparedEvaluator[Value],
        ) -> EvaluatorResult:
            return await evaluator(value)

        return cls(
            evaluations=tuple(
                await execute_concurrently(
                    execute,
                    (evaluator, *evaluators),
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
        """
        Merge multiple scenario results into one.

        Parameters
        ----------
        result : Self
            First result to merge
        *results : Self
            Additional results to merge
        meta : Meta | MetaValues | None, optional
            Additional metadata to include in merged result

        Returns
        -------
        Self
            New result containing all evaluations from input results
        """
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


class EvaluatorScenarioResult(DataModel):
    """
    Result of running an evaluator scenario.

    Contains evaluation results from running a named scenario, including
    pass/fail status and performance metrics across all evaluations.

    Attributes
    ----------
    scenario : str
        Name of the evaluated scenario
    evaluations : Sequence[EvaluatorResult]
        Results from all evaluators in the scenario
    meta : Meta
        Additional metadata for the scenario
    """

    scenario: str = Field(
        description="Name of the evaluated scenario",
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
        """
        Check if all evaluations in the scenario passed.

        Returns
        -------
        bool
            True if all evaluations passed, False otherwise.
            Empty evaluations return False.
        """
        # empty evaluations is equivalent of failure
        return len(self.evaluations) > 0 and all(case.passed for case in self.evaluations)

    def report(
        self,
        *,
        detailed: bool = True,
        include_passed: bool = True,
    ) -> str:
        """
        Generate a human-readable report of the scenario results.

        Parameters
        ----------
        detailed : bool, optional
            If True, include full details in XML format. If False, return
            brief summary, by default True
        include_passed : bool, optional
            If True, include passed evaluations in the report. If False,
            only show failed evaluations, by default True

        Returns
        -------
        str
            Formatted report string
        """
        if not self.evaluations:
            return f"{self.scenario} has no evaluations"

        status: str = "passed" if self.passed else "failed"
        results_report: str = "\n".join(
            result.report(detailed=detailed)
            for result in self.evaluations
            if include_passed or not result.passed
        )

        if not detailed:
            if results_report:
                return f"{self.scenario}: {status} ({self.performance}%)\n{results_report}"

            else:
                return f"{self.scenario}: {status} ({self.performance}%)"

        if results_report:
            return (
                f"<evaluator_scenario name='{self.scenario}' status='{status}'"
                f" performance='{self.performance:.2f}%'>"
                f"\n<evaluators>\n{results_report}\n</evaluators>"
                "\n</evaluator_scenario>"
            )

        else:
            return (
                f"<evaluator_scenario name='{self.scenario}' status='{status}'"
                f" performance='{self.performance:.2f}%'/>"
            )

    @property
    def performance(self) -> float:
        """
        Calculate average performance across all evaluations.

        Returns
        -------
        float
            Average performance percentage (0-100) across all evaluations.
            Returns 0.0 if no evaluations.
        """
        if not self.evaluations:
            return 0.0

        score: float = 0.0
        for evaluation in self.evaluations:
            score += evaluation.performance

        return score / len(self.evaluations)


@runtime_checkable
class PreparedEvaluatorScenario[Value](Protocol):
    """
    Protocol for evaluator scenarios with pre-configured arguments.

    A prepared evaluator scenario has all arguments except the value to evaluate
    already bound, simplifying the evaluation interface.
    """

    async def __call__(
        self,
        value: Value,
        /,
    ) -> EvaluatorScenarioResult: ...


@runtime_checkable
class EvaluatorScenarioDefinition[Value, **Args](Protocol):
    """
    Protocol for evaluator scenario function definitions.

    Defines the interface for functions that can run multiple evaluations
    on a value and return either a scenario result or sequence of evaluator results.
    """

    @property
    def __name__(self) -> str: ...

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluationScenarioResult | Sequence[EvaluatorResult]: ...


class EvaluatorScenario[Value, **Args]:
    """
    Configurable evaluator scenario for running multiple evaluations.

    An EvaluatorScenario wraps a scenario function with configuration like name,
    state, and metadata. It provides methods to transform inputs and prepare
    scenarios with partial arguments.

    Attributes
    ----------
    name : str
        Identifier for the scenario
    meta : Meta
        Additional metadata for the scenario
    """

    __slots__ = (
        "_definition",
        "_state",
        "meta",
        "name",
    )

    def __init__(
        self,
        name: str,
        definition: EvaluatorScenarioDefinition[Value, Args],
        state: Collection[State],
        meta: Meta,
    ) -> None:
        assert "\n" not in name  # nosec: B101

        self.name: str
        object.__setattr__(
            self,
            "name",
            name.lower().replace(" ", "_"),
        )
        self._definition: EvaluatorScenarioDefinition[Value, Args]
        object.__setattr__(
            self,
            "_definition",
            definition,
        )
        self._state: Collection[State]
        object.__setattr__(
            self,
            "_state",
            state,
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
    ) -> PreparedEvaluatorScenario[Value]:
        """
        Create a prepared evaluator scenario with bound arguments.

        Parameters
        ----------
        *args : Args.args
            Positional arguments to bind
        **kwargs : Args.kwargs
            Keyword arguments to bind

        Returns
        -------
        PreparedEvaluatorScenario[Value]
            Scenario with pre-bound arguments
        """

        async def evaluate(
            value: Value,
        ) -> EvaluatorScenarioResult:
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
        """
        Create a copy with a different name.

        Parameters
        ----------
        name : str
            New name for the scenario

        Returns
        -------
        Self
            New scenario instance with updated name
        """
        return self.__class__(
            name=name,
            definition=self._definition,
            state=self._state,
            meta=self.meta,
        )

    def with_state(
        self,
        state: State,
        /,
        *states: State,
    ) -> Self:
        """
        Create a copy with additional state.

        Parameters
        ----------
        state : State
            First state object to add
        *states : State
            Additional state objects to add

        Returns
        -------
        Self
            New scenario instance with extended state
        """
        return self.__class__(
            name=self.name,
            definition=self._definition,
            state=(*self._state, state, *states),
            meta=self.meta,
        )

    def with_meta(
        self,
        meta: Meta | MetaValues,
        /,
    ) -> Self:
        """
        Create a copy with merged metadata.

        Parameters
        ----------
        meta : Meta | MetaValues
            Metadata to merge with existing metadata

        Returns
        -------
        Self
            New scenario instance with merged metadata
        """
        return self.__class__(
            name=self.name,
            definition=self._definition,
            state=self._state,
            meta=self.meta.merged_with(meta),
        )

    def contra_map[Mapped](
        self,
        mapping: Callable[[Mapped], Value] | AttributePath[Mapped, Value] | Value,
        /,
    ) -> "EvaluatorScenario[Mapped, Args]":
        """
        Transform the input value before evaluation.

        Parameters
        ----------
        mapping : Callable[[Mapped], Value] | AttributePath[Mapped, Value] | Value
            Function or attribute path to transform input values

        Returns
        -------
        EvaluatorScenario[Mapped, Args]
            New scenario that transforms inputs before evaluation
        """
        mapper: Callable[[Mapped], Value]
        if isinstance(mapping, AttributePath):
            mapper = cast(AttributePath[Mapped, Value], mapping)

        else:
            assert isinstance(mapping, Callable)  # nosec: B101
            mapper = mapping

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

        return EvaluatorScenario[Mapped, Args](
            name=self.name,
            definition=evaluation,
            state=self._state,
            meta=self.meta,
        )

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorScenarioResult:
        async with ctx.scope(f"evaluation.scenario.{self.name}", *self._state):
            result: EvaluatorScenarioResult = await self._evaluate(
                value,
                *args,
                **kwargs,
            )

        ctx.record(
            metric=f"evaluation.scenario.{result.scenario}.performance",
            value=result.performance,
            unit="%",
            kind="histogram",
            attributes={
                "evaluation.passed": result.passed,
                "evaluation.scenario.evaluations": [
                    evaluation.evaluator for evaluation in result.evaluations
                ],
            },
        )
        return result

    async def _evaluate(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorScenarioResult:
        result: EvaluationScenarioResult | Sequence[EvaluatorResult]
        try:
            result = await self._definition(
                value,
                *args,
                **kwargs,
            )

        except Exception as exc:
            ctx.log_error(
                f"Evaluator scenario `{self.name}` failed due to an error",
                exception=exc,
            )
            return EvaluatorScenarioResult(
                scenario=self.name,
                evaluations=(),
                meta=self.meta.merged_with(
                    {
                        "exception": str(type(exc)),
                        "error": str(exc),
                    }
                ),
            )

        if isinstance(result, EvaluationScenarioResult):
            return EvaluatorScenarioResult(
                scenario=self.name,
                evaluations=result.evaluations,
                meta=self.meta,
            )

        else:
            return EvaluatorScenarioResult(
                scenario=self.name,
                evaluations=result,
                meta=self.meta,
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
def evaluator_scenario[Value, **Args](
    definition: EvaluatorScenarioDefinition[Value, Args],
    /,
) -> EvaluatorScenario[Value, Args]: ...


@overload
def evaluator_scenario[Value, **Args](
    *,
    name: str | None = None,
    state: Collection[State] = (),
    meta: Meta | MetaValues | None = None,
) -> Callable[
    [EvaluatorScenarioDefinition[Value, Args]],
    EvaluatorScenario[Value, Args],
]: ...


def evaluator_scenario[Value, **Args](  # pyright: ignore[reportInconsistentOverload] - this seems to be pyright false positive/error
    definition: EvaluatorScenarioDefinition[Value, Args] | None = None,
    /,
    *,
    name: str | None = None,
    state: Collection[State] = (),
    meta: Meta | MetaValues | None = None,
) -> (
    Callable[
        [EvaluatorScenarioDefinition[Value, Args]],
        EvaluatorScenario[Value, Args],
    ]
    | EvaluatorScenario[Value, Args]
):
    """
    Create or decorate an evaluator scenario function.

    Can be used as a decorator or called directly to create an EvaluatorScenario
    from a scenario function.

    Parameters
    ----------
    definition : EvaluatorScenarioDefinition[Value, Args] | None, optional
        The scenario function. If None, returns a decorator
    name : str | None, optional
        Name for the scenario. If None, uses function's __name__
    state : Collection[State], optional
        State objects to include in evaluation context
    meta : Meta | MetaValues | None, optional
        Metadata for the scenario

    Returns
    -------
    Callable[[EvaluatorScenarioDefinition[Value, Args]], EvaluatorScenario[Value, Args]] | EvaluatorScenario[Value, Args]
        Either a decorator (if definition is None) or an EvaluatorScenario instance

    Examples
    --------
    As a decorator:
    >>> @evaluator_scenario
    ... async def my_scenario(value: str) -> Sequence[EvaluatorResult]:
    ...     return [await evaluator1(value), await evaluator2(value)]

    Direct call:
    >>> my_scenario = evaluator_scenario(some_function, name="custom")
    """  # noqa: E501

    def wrap(
        definition: EvaluatorScenarioDefinition[Value, Args],
    ) -> EvaluatorScenario[Value, Args]:
        return EvaluatorScenario(
            name=name or definition.__name__,
            definition=definition,
            state=state,
            meta=Meta.of(meta),
        )

    if definition:
        return wrap(definition)

    else:
        return wrap
