import re
from collections.abc import Callable, Collection, Sequence
from typing import Annotated, Protocol, Self, cast, overload, runtime_checkable

from haiway import (
    META_EMPTY,
    AttributePath,
    Description,
    Immutable,
    Meta,
    MetaValues,
    State,
    Validator,
    concurrently,
    ctx,
)

from draive.evaluation.score import EvaluationScore
from draive.evaluation.value import (
    EvaluationScoreValue,
    evaluation_score_value,
    evaluation_score_verifier,
)
from draive.parameters import DataModel

__all__ = (
    "Evaluator",
    "EvaluatorDefinition",
    "EvaluatorResult",
    "PreparedEvaluator",
    "evaluator",
)


class EvaluatorResult(DataModel):
    """
    Result from running an evaluator on a value.

    Contains the evaluation score, threshold for passing, and metadata about
    which evaluator was used. Can determine if the evaluation passed based
    on the score meeting the threshold.
    """

    @classmethod
    def of(
        cls,
        evaluator: str,
        /,
        *,
        score: EvaluationScore | EvaluationScoreValue,
        threshold: EvaluationScoreValue,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Create an EvaluatorResult.

        Parameters
        ----------
        evaluator : str
            Name of the evaluator that produced this result
        score : EvaluationResult | EvaluationScore | EvaluationScoreValue
            The evaluation score in various formats
        threshold : EvaluationScoreValue
            The minimum score value required to pass
        meta : Meta | MetaValues | None, optional
            Additional metadata for the result

        Returns
        -------
        Self
            New EvaluatorResult instance
        """
        evaluation_score: EvaluationScoreValue
        evaluation_meta: Meta
        if isinstance(score, EvaluationScore):
            evaluation_score = score.value
            evaluation_meta = score.meta.merged_with(meta)

        else:
            evaluation_score = evaluation_score_value(score)
            evaluation_meta = Meta.of(meta)

        return cls(
            evaluator=evaluator,
            score=evaluation_score,
            threshold=evaluation_score_value(threshold),
            meta=evaluation_meta,
        )

    evaluator: Annotated[
        str,
        Description("Name of the evaluator"),
    ]
    score: Annotated[
        float,
        Description("Evaluation score value"),
        Validator(evaluation_score_verifier),
    ]
    threshold: Annotated[
        float,
        Description(
            "Score threshold required to pass evaluation, a value between 0 (min) and 1 (max)"
        ),
        Validator(evaluation_score_verifier),
    ]
    meta: Annotated[
        Meta,
        Description("Additional evaluation metadata"),
    ] = META_EMPTY

    @property
    def passed(self) -> bool:
        """
        Check if the evaluation passed.

        Returns
        -------
        bool
            True if score value meets or exceeds threshold
        """
        return self.score >= self.threshold

    def report(
        self,
        detailed: bool = True,
    ) -> str:
        """
        Generate a human-readable report of the evaluation result.

        Parameters
        ----------
        detailed : bool, optional
            If True, include full details in XML format. If False, return
            a brief summary line, by default True

        Returns
        -------
        str
            Formatted report string
        """
        status: str = "passed" if self.passed else "failed"
        if not detailed:
            return f"{self.evaluator}: {status} ({self.performance:.2f}%)"

        meta: list[str] = []
        for key, value in self.meta.items():
            meta.append(f"<{key}>{value}</{key}>")

        return (
            f"<evaluator name='{self.evaluator}' status='{status}'"
            f" performance='{self.performance:.2f}%'>"
            f"\n<score>{self.score:.3f}</score>"
            f"\n<threshold>{self.threshold:.3f}</threshold>"
            f"\n<meta>\n{'\n'.join(meta)}\n</meta>"
            "\n</evaluator>"
        )

    @property
    def performance(self) -> float:
        """
        Calculate performance as a percentage.

        Returns
        -------
        float
            Performance percentage, may go above 100% if score is higher than threshold
        """
        if self.threshold <= 0:
            return self.score * 100.0

        return (self.score / self.threshold) * 100.0

    def __gt__(
        self,
        other: Self,
    ) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score > other.score

    def __ge__(
        self,
        other: Self,
    ) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score >= other.score

    def __lt__(
        self,
        other: Self,
    ) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score < other.score

    def __le__(
        self,
        other: Self,
    ) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score <= other.score

    def __eq__(
        self,
        other: Self,
    ) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score == other.score

    def __hash__(self) -> int:  # explicitly using super to silence warnings
        return hash((self.evaluator, self.score, self.threshold))


@runtime_checkable
class EvaluatorDefinition[Value, **Args](Protocol):
    """
    Protocol for evaluator function definitions.

    Defines the interface for functions that can evaluate a value and return
    a score. Evaluators must have a __name__ property and be callable with
    the value to evaluate plus additional arguments.
    """

    @property
    def __name__(self) -> str: ...

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluationScore | EvaluationScoreValue: ...


@runtime_checkable
class PreparedEvaluator[Value](Protocol):
    """
    Protocol for evaluators with pre-configured arguments.

    A prepared evaluator has all arguments except the value to evaluate
    already bound, simplifying the evaluation interface.
    """

    async def __call__(
        self,
        value: Value,
        /,
    ) -> EvaluatorResult: ...


class Evaluator[Value, **Args](Immutable):
    """
    Configurable evaluator for assessing values against criteria.

    An Evaluator wraps an evaluation function with configuration like name,
    threshold, state, and metadata. It provides methods to compose evaluators,
    transform inputs, and prepare evaluators with partial arguments.

    The evaluation uses "evaluator.evaluator_name" context where evaluator_name is actual name of
    the evaluator to execute evaluation.

    Attributes
    ----------
    name : str
        Identifier for the evaluator
    threshold : float
        Minimum score (0-1) required for evaluation to pass
    meta : Meta
        Additional metadata for the evaluator
    """

    @staticmethod
    def lowest(
        evaluator: PreparedEvaluator[Value],
        /,
        *evaluators: PreparedEvaluator[Value],
        concurrent_tasks: int = 2,
    ) -> PreparedEvaluator[Value]:
        """
        Create an evaluator that returns the lowest performing result.

        Parameters
        ----------
        evaluator : PreparedEvaluator[Value]
            First evaluator to run
        *evaluators : PreparedEvaluator[Value]
            Additional evaluators to run
        concurrent_tasks: int
            Number of concurrently executed evaluators

        Returns
        -------
        PreparedEvaluator[Value]
            Evaluator that returns the result with lowest performance percentage
        """

        async def evaluate(
            value: Value,
        ) -> EvaluatorResult:
            # Placeholder for the lowest result
            lowest: EvaluatorResult = EvaluatorResult(
                evaluator="lowest",
                score=1.0,
                threshold=0,
                meta=META_EMPTY,
            )

            for result in await concurrently(
                (evaluator(value), *(evaluator(value) for evaluator in evaluators)),
                concurrent_tasks=concurrent_tasks,
                return_exceptions=False,
            ):
                if result.performance <= lowest.performance:
                    lowest = result

            return lowest

        return evaluate

    @staticmethod
    def highest(
        evaluator: PreparedEvaluator[Value],
        /,
        *evaluators: PreparedEvaluator[Value],
        concurrent_tasks: int = 2,
    ) -> PreparedEvaluator[Value]:
        """
        Create an evaluator that returns the highest performing result.

        Parameters
        ----------
        evaluator : PreparedEvaluator[Value]
            First evaluator to run
        *evaluators : PreparedEvaluator[Value]
            Additional evaluators to run
        concurrent_tasks: int
            Number of concurrently executed evaluators

        Returns
        -------
        PreparedEvaluator[Value]
            Evaluator that returns the result with highest performance percentage
        """

        async def evaluate(
            value: Value,
        ) -> EvaluatorResult:
            # Placeholder for the highest result
            highest: EvaluatorResult = EvaluatorResult(
                evaluator="highest",
                score=0.0,
                threshold=1,
                meta=META_EMPTY,
            )

            for result in await concurrently(
                (evaluator(value), *(evaluator(value) for evaluator in evaluators)),
                concurrent_tasks=concurrent_tasks,
                return_exceptions=False,
            ):
                if result.performance >= highest.performance:
                    highest = result

            return highest

        return evaluate

    @staticmethod
    def average(
        evaluator: PreparedEvaluator[Value],
        /,
        *evaluators: PreparedEvaluator[Value],
        threshold: EvaluationScoreValue,
        concurrent_tasks: int = 2,
    ) -> PreparedEvaluator[Value]:
        """
        Create an evaluator that returns the average scoring result.

        Parameters
        ----------
        evaluator : PreparedEvaluator[Value]
            First evaluator to run
        *evaluators : PreparedEvaluator[Value]
            Additional evaluators to run
        threshold: EvaluationScoreValue
            Combined result threshold replacing combined thresholds
        concurrent_tasks: int
            Number of concurrently executed evaluators

        Returns
        -------
        PreparedEvaluator[Value]
            Evaluator that returns the result with average score value
        """

        async def evaluate(
            value: Value,
        ) -> EvaluatorResult:
            scores: Sequence[float] = [
                result.score
                for result in await concurrently(
                    (evaluator(value), *(evaluator(value) for evaluator in evaluators)),
                    concurrent_tasks=concurrent_tasks,
                    return_exceptions=False,
                )
            ]

            return EvaluatorResult(
                evaluator="average",
                score=sum(scores) / len(scores),
                threshold=evaluation_score_value(threshold),
                meta=META_EMPTY,
            )

        return evaluate

    name: str
    threshold: float
    meta: Meta
    _definition: EvaluatorDefinition[Value, Args]
    _state: Collection[State]

    def __init__(
        self,
        name: str,
        definition: EvaluatorDefinition[Value, Args],
        threshold: float,
        state: Collection[State],
        meta: Meta,
    ) -> None:
        assert (  # nosec: B101
            threshold is None or 0 <= threshold <= 1
        ), "Evaluator threshold has to be a value between 0 and 1"
        assert re.match(  # nosec: B101
            r"^[a-z][a-z0-9_]*$", name.lower()
        ), "Evaluator name should follow snake case rules"

        object.__setattr__(
            self,
            "_definition",
            definition,
        )
        object.__setattr__(
            self,
            "name",
            name.lower().replace(" ", "_"),
        )
        object.__setattr__(
            self,
            "threshold",
            threshold,
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
            New name for the evaluator

        Returns
        -------
        Self
            New evaluator instance with updated name
        """
        return self.__class__(
            name=name,
            definition=self._definition,
            threshold=self.threshold,
            state=self._state,
            meta=self.meta,
        )

    def with_threshold(
        self,
        value: EvaluationScoreValue,
        /,
    ) -> Self:
        """
        Create a copy with a different threshold.

        Parameters
        ----------
        value : EvaluationScoreValue
            New threshold value

        Returns
        -------
        Self
            New evaluator instance with updated threshold
        """
        return self.__class__(
            name=self.name,
            definition=self._definition,
            threshold=evaluation_score_value(value),
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
            New evaluator instance with extended state
        """
        return self.__class__(
            name=self.name,
            definition=self._definition,
            threshold=self.threshold,
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
            New evaluator instance with merged metadata
        """
        return self.__class__(
            name=self.name,
            definition=self._definition,
            threshold=self.threshold,
            state=self._state,
            meta=self.meta.merged_with(meta),
        )

    def prepared(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> PreparedEvaluator[Value]:
        """
        Create a prepared evaluator with bound arguments.

        Parameters
        ----------
        *args : Args.args
            Positional arguments to bind
        **kwargs : Args.kwargs
            Keyword arguments to bind

        Returns
        -------
        PreparedEvaluator[Value]
            Evaluator with pre-bound arguments
        """

        async def evaluate(
            value: Value,
        ) -> EvaluatorResult:
            return await self(
                value,
                *args,
                **kwargs,
            )

        return evaluate

    def contra_map[Mapped](
        self,
        mapping: Callable[[Mapped], Value] | AttributePath[Mapped, Value] | Value,
        /,
    ) -> "Evaluator[Mapped, Args]":
        """
        Transform the input value before evaluation.

        Parameters
        ----------
        mapping : Callable[[Mapped], Value] | AttributePath[Mapped, Value] | Value
            Function or attribute path to transform input values

        Returns
        -------
        Evaluator[Mapped, Args]
            New evaluator that transforms inputs before evaluation
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
        ) -> EvaluationScore | EvaluationScoreValue:
            return await self._definition(
                mapper(value),
                *args,
                **kwargs,
            )

        return Evaluator[Mapped, Args](
            name=self.name,
            definition=evaluation,
            threshold=self.threshold,
            state=self._state,
            meta=self.meta,
        )

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorResult:
        async with ctx.scope(f"evaluator.{self.name}", *self._state):
            result: EvaluatorResult = await self._evaluate(
                value,
                *args,
                **kwargs,
            )

            ctx.record_info(
                metric=f"evaluator.{result.evaluator}.performance",
                value=result.performance,
                unit="%",
                kind="histogram",
                attributes={
                    "passed": result.passed,
                    "threshold": result.threshold,
                    "score": result.score,
                },
            )
            return result

    async def _evaluate(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorResult:
        result: EvaluationScore | EvaluationScoreValue
        try:
            result = await self._definition(
                value,
                *args,
                **kwargs,
            )

        except Exception as exc:
            ctx.log_error(
                f"Evaluator `{self.name}` failed due to an error",
                exception=exc,
            )
            result = EvaluationScore.of(
                0.0,
                meta={
                    "comment": "Failed due to the evaluator issue",
                    "exception": str(type(exc)),
                    "error": str(exc),
                },
            )

        return EvaluatorResult.of(
            self.name,
            score=result,
            threshold=self.threshold,
            meta=self.meta,
        )


@overload
def evaluator[Value, **Args](
    definition: EvaluatorDefinition[Value, Args],
    /,
) -> Evaluator[Value, Args]: ...


@overload
def evaluator[Value, **Args](
    *,
    name: str | None = None,
    threshold: EvaluationScoreValue = 1,
    state: Collection[State] = (),
    meta: Meta | MetaValues | None = None,
) -> Callable[
    [EvaluatorDefinition[Value, Args]],
    Evaluator[Value, Args],
]: ...


def evaluator[Value, **Args](
    evaluation: EvaluatorDefinition[Value, Args] | None = None,
    /,
    *,
    name: str | None = None,
    threshold: EvaluationScoreValue = 1,
    state: Collection[State] = (),
    meta: Meta | MetaValues | None = None,
) -> (
    Callable[
        [EvaluatorDefinition[Value, Args]],
        Evaluator[Value, Args],
    ]
    | Evaluator[Value, Args]
):
    """
    Create or decorate an evaluator function.

    Can be used as a decorator or called directly to create an Evaluator
    from an evaluation function.

    Parameters
    ----------
    evaluation : EvaluatorDefinition[Value, Args] | None, optional
        The evaluation function. If None, returns a decorator
    name : str | None, optional
        Name for the evaluator. If None, uses function's __name__
    threshold : EvaluationScoreValue, optional
        Minimum score to pass evaluation, by default 1
    state : Collection[State], optional
        State objects to include in evaluation context
    meta : Meta | MetaValues | None, optional
        Metadata for the evaluator

    Returns
    -------
    Callable[[EvaluatorDefinition[Value, Args]], Evaluator[Value, Args]] | Evaluator[Value, Args]
        Either a decorator (if evaluation is None) or an Evaluator instance

    Examples
    --------
    >>> @evaluator(threshold=0.8)
    ... async def my_evaluator(value: str) -> float:
    ...     return 0.9 if len(value) > 10 else 0.5
    """

    def wrap(
        definition: EvaluatorDefinition[Value, Args],
    ) -> Evaluator[Value, Args]:
        return Evaluator(
            name=name or definition.__name__,
            definition=definition,
            threshold=evaluation_score_value(threshold),
            state=state,
            meta=Meta.of(meta),
        )

    if evaluation:
        return wrap(evaluation)

    else:
        return wrap
