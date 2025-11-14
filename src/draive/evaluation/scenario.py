import re
from collections.abc import Callable, Sequence
from typing import Annotated, Protocol, Self, cast, overload, runtime_checkable

from haiway import (
    AttributePath,
    Description,
    Immutable,
    Meta,
    MetaValues,
    State,
    ctx,
)

from draive.evaluation.evaluator import EvaluatorResult
from draive.parameters import DataModel

__all__ = (
    "EvaluatorScenario",
    "EvaluatorScenarioDefinition",
    "EvaluatorScenarioResult",
    "evaluator_scenario",
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
    results : Sequence[EvaluatorResult]
        Results from all evaluators in the scenario
    """

    scenario: Annotated[
        str,
        Description("Name of the evaluated scenario"),
    ]
    results: Annotated[
        Sequence[EvaluatorResult],
        Description("Scenario evaluation results"),
    ]

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
        return len(self.results) > 0 and all(result.passed for result in self.results)

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
        if not self.results:
            return f"{self.scenario} has no results"

        status: str = "passed" if self.passed else "failed"
        results_report: str = "\n".join(
            result.report(detailed=detailed)
            for result in self.results
            if include_passed or not result.passed
        )

        if not detailed:
            if results_report:
                return f"{self.scenario}: {status} ({self.performance:.2f}%)\n{results_report}"

            else:
                return f"{self.scenario}: {status} ({self.performance:.2f}%)"

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
        if not self.results:
            return 0.0

        score: float = 0.0
        for result in self.results:
            score += min(100.0, result.performance)  # normalize results

        return score / len(self.results)


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
    ) -> Sequence[EvaluatorResult]: ...


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


class EvaluatorScenario[Value, **Args](Immutable):
    """
    Configurable evaluator scenario for running multiple evaluations.

    An EvaluatorScenario wraps a scenario function with configuration like name,
    state, and metadata. It provides methods to transform inputs and prepare
    scenarios with partial arguments.

    The evaluation uses "evaluator.scenario.scenario_name" context where scenario_name is
    an actual name of the evaluator scenario.

    Attributes
    ----------
    name : str
        Identifier for the scenario
    meta : Meta
        Additional metadata for the scenario
    """

    name: str
    meta: Meta
    _definition: EvaluatorScenarioDefinition[Value, Args]
    _state: Sequence[State]

    def __init__(
        self,
        name: str,
        definition: EvaluatorScenarioDefinition[Value, Args],
        state: Sequence[State],
        meta: Meta,
    ) -> None:
        assert re.match(  # nosec: B101
            r"^[a-z][a-z0-9_]*$", name.lower()
        ), "Evaluator name should follow snake case rules"

        object.__setattr__(
            self,
            "name",
            name.lower().replace(" ", "_"),
        )
        object.__setattr__(
            self,
            "_definition",
            definition,
        )
        object.__setattr__(
            self,
            "_state",
            state,
        )
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
            mapper = cast(Callable[[Mapped], Value], mapping)

        async def evaluation(
            value: Mapped,
            /,
            *args: Args.args,
            **kwargs: Args.kwargs,
        ) -> Sequence[EvaluatorResult]:
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
        async with ctx.scope(f"evaluator.scenario.{self.name}", *self._state):
            result: EvaluatorScenarioResult = await self._evaluate(
                value,
                *args,
                **kwargs,
            )

            ctx.record_info(
                metric=f"evaluator.scenario.{result.scenario}.performance",
                value=result.performance,
                unit="%",
                kind="histogram",
                attributes={
                    "passed": result.passed,
                    "evaluators": [result.evaluator for result in result.results],
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
        results: Sequence[EvaluatorResult]
        try:
            results = await self._definition(
                value,
                *args,
                **kwargs,
            )

        except Exception as exc:
            ctx.log_error(
                f"Evaluator scenario `{self.name}` failed due to an error",
                exception=exc,
            )
            results = ()

        return EvaluatorScenarioResult(
            scenario=self.name,
            results=results,
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
    state: Sequence[State] = (),
    meta: Meta | MetaValues | None = None,
) -> Callable[
    [EvaluatorScenarioDefinition[Value, Args]],
    EvaluatorScenario[Value, Args],
]: ...


def evaluator_scenario[Value, **Args](
    definition: EvaluatorScenarioDefinition[Value, Args] | None = None,
    /,
    *,
    name: str | None = None,
    state: Sequence[State] = (),
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
    state : Sequence[State], optional
        State objects to include in evaluation context
    meta : Meta | MetaValues | None, optional
        Metadata for the scenario

    Returns
    -------
    Callable[[EvaluatorScenarioDefinition[Value, Args]], EvaluatorScenario[Value, Args]] | EvaluatorScenario[Value, Args]
        Either a decorator (if definition is None) or an EvaluatorScenario instance

    Examples
    --------
    >>> @evaluator_scenario
    ... async def my_scenario(value: str) -> Sequence[EvaluatorResult]:
    ...     return [await evaluator1(value), await evaluator2(value)]
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
