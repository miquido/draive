from asyncio import gather
from collections.abc import Callable
from typing import Protocol, Self, cast, final, overload, runtime_checkable

from haiway import AttributePath, ScopeContext, ctx, freeze

from draive.commons import META_EMPTY, Meta
from draive.evaluation.score import EvaluationScore
from draive.evaluation.value import EvaluationScoreValue, evaluation_score_value
from draive.parameters import DataModel, Field

__all__ = (
    "Evaluator",
    "EvaluatorDefinition",
    "EvaluatorResult",
    "PreparedEvaluator",
    "evaluator",
)


def _verifier(
    value: float,
) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"Threshold has to have a value between 0 and 1, received: {value}")


class EvaluatorResult(DataModel):
    @classmethod
    def of(
        cls,
        evaluator: str,
        /,
        *,
        score: EvaluationScoreValue,
        score_comment: str | None = None,
        threshold: EvaluationScoreValue,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            evaluator=evaluator,
            score=EvaluationScore.of(
                score,
                comment=score_comment,
            ),
            threshold=evaluation_score_value(threshold),
            meta=meta if meta is not None else META_EMPTY,
        )

    evaluator: str = Field(
        description="Name of evaluator",
    )
    score: EvaluationScore = Field(
        description="Evaluation score",
    )
    threshold: float = Field(
        description="Score threshold required to pass evaluation, "
        "a value between 0 (min) and 1 (max)",
        verifier=_verifier,
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )

    @property
    def passed(self) -> bool:
        return self.score.value >= self.threshold

    def report(
        self,
        include_details: bool = True,
    ) -> str:
        if include_details:
            meta_values: str = (
                f"\n{'\n'.join(f'{key}: {value}' for key, value in self.meta.items())}"
                if self.meta
                else "N/A"
            )

            return (
                f"{self.evaluator} {'passed' if self.passed else 'failed'}"
                f" with score {self.score.value},"
                f" required {self.threshold},"
                f" comment: {f"'{self.score.comment}'" or 'N/A'}"
                f" meta:\n{meta_values}"
            )

        else:
            return (
                f"{self.evaluator} {'passed' if self.passed else 'failed'}"
                f" comment: {f"'{self.score.comment}'" or 'N/A'}"
            )

    def __gt__(self, other: Self) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score > other.score

    def __ge__(self, other: Self) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score >= other.score

    def __lt__(self, other: Self) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score < other.score

    def __le__(self, other: Self) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score <= other.score

    def __eq__(self, other: Self) -> bool:
        assert isinstance(other, self.__class__)  # nosec: B101
        if self.evaluator != other.evaluator or self.threshold != other.threshold:
            raise ValueError("Can't compare different evaluator results")

        return self.score == other.score


class EvaluationResult(DataModel):
    @classmethod
    async def of(
        cls,
        score: EvaluationScore | float | bool,
        /,
        meta: Meta | None = None,
    ) -> Self:
        evaluation_score: EvaluationScore
        match score:
            case EvaluationScore() as score:
                evaluation_score = score

            case float() as value:
                evaluation_score = EvaluationScore(value=value)

            case passed:
                evaluation_score = EvaluationScore(value=1.0 if passed else 0.0)

        return cls(
            score=evaluation_score,
            meta=meta if meta is not None else META_EMPTY,
        )

    score: EvaluationScore = Field(
        description="Evaluation score",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )


@runtime_checkable
class EvaluatorDefinition[Value, **Args](Protocol):
    @property
    def __name__(self) -> str: ...

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluationResult | EvaluationScore | float | bool: ...


@runtime_checkable
class PreparedEvaluator[Value](Protocol):
    async def __call__(
        self,
        value: Value,
        /,
    ) -> EvaluatorResult: ...


@final
class Evaluator[Value, **Args]:
    @staticmethod
    def lowest(
        evaluators: PreparedEvaluator[Value],
        /,
        *_evaluators: PreparedEvaluator[Value],
    ) -> PreparedEvaluator[Value]:
        async def evaluate(
            value: Value,
        ) -> EvaluatorResult:
            # Placeholder for the lowest result
            lowest: EvaluatorResult = EvaluatorResult(
                evaluator="lowest",
                score=EvaluationScore(value=1),
                threshold=0,
                meta=META_EMPTY,
            )

            for result in await gather(
                evaluators(value),
                *[evaluator(value) for evaluator in _evaluators],
            ):
                if result.score <= lowest.score:
                    lowest = result

            return lowest

        return evaluate

    @staticmethod
    def highest(
        evaluators: PreparedEvaluator[Value],
        /,
        *_evaluators: PreparedEvaluator[Value],
    ) -> PreparedEvaluator[Value]:
        async def evaluate(
            value: Value,
        ) -> EvaluatorResult:
            # Placeholder for the highest result
            highest: EvaluatorResult = EvaluatorResult(
                evaluator="highest",
                score=EvaluationScore(value=0),
                threshold=1,
                meta=META_EMPTY,
            )

            for result in await gather(
                evaluators(value),
                *[evaluator(value) for evaluator in _evaluators],
            ):
                if result.score >= highest.score:
                    highest = result

            return highest

        return evaluate

    def __init__(
        self,
        name: str,
        definition: EvaluatorDefinition[Value, Args],
        threshold: float | None,
        execution_context: ScopeContext | None,
        meta: Meta | None,
    ) -> None:
        assert (  # nosec: B101
            threshold is None or 0 <= threshold <= 1
        ), "Evaluation threshold has to be between 0 and 1"
        assert "\n" not in name  # nosec: B101

        self._definition: EvaluatorDefinition[Value, Args] = definition
        self._execution_context: ScopeContext | None = execution_context
        self.name: str = name.lower().replace(" ", "_")
        self.threshold: float = 1 if threshold is None else threshold
        self.meta: Meta = meta if meta is not None else META_EMPTY

        freeze(self)

    def with_name(
        self,
        name: str,
        /,
    ) -> Self:
        return self.__class__(
            name=name,
            definition=self._definition,
            threshold=self.threshold,
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
            threshold=self.threshold,
            execution_context=context,
            meta=self.meta,
        )

    def with_threshold(
        self,
        value: EvaluationScoreValue,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            threshold=evaluation_score_value(value),
            execution_context=self._execution_context,
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
            threshold=self.threshold,
            execution_context=self._execution_context,
            meta={**self.meta, **meta} if self.meta else meta,
        )

    def prepared(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> PreparedEvaluator[Value]:
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
        ) -> EvaluationResult | EvaluationScore | float | bool:
            return await self._definition(
                mapper(value),
                *args,
                **kwargs,
            )

        return Evaluator[Mapped, Args](
            name=self.name,
            definition=evaluation,
            threshold=self.threshold,
            execution_context=self._execution_context,
            meta=self.meta,
        )

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorResult:
        result: EvaluatorResult
        if context := self._execution_context:
            async with context:
                result = await self._evaluate(
                    value,
                    *args,
                    **kwargs,
                )

        else:
            async with ctx.scope(f"evaluator.{self.name}"):
                result = await self._evaluate(
                    value,
                    *args,
                    **kwargs,
                )

        ctx.record(
            metric=f"evaluator.{result.evaluator}.score",
            value=result.score.value,
            unit="%",
            attributes={
                "evaluation.threshold": result.threshold,
                "evaluation.passed": result.passed,
                "evaluation.score.comment": result.score.comment,
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
        evaluation_score: EvaluationScore
        evaluation_meta: Meta
        try:
            match await self._definition(
                value,
                *args,
                **kwargs,
            ):
                case EvaluationResult() as result:
                    evaluation_score = result.score
                    evaluation_meta = result.meta

                case EvaluationScore() as score:
                    evaluation_score = score
                    evaluation_meta = META_EMPTY

                case float() as score_value:
                    evaluation_score = EvaluationScore(value=score_value)
                    evaluation_meta = META_EMPTY

                case passed:
                    evaluation_score = EvaluationScore(value=1 if passed else 0)
                    evaluation_meta = META_EMPTY

        except Exception as exc:
            ctx.log_error(
                f"Evaluator `{self.name}` failed, using `0` score fallback result",
                exception=exc,
            )
            evaluation_score = EvaluationScore(
                value=0,
                comment="Evaluation failed",
            )
            evaluation_meta = {"exception": str(exc)}

        result_meta: Meta
        if self.meta:
            if evaluation_meta:
                result_meta = {**self.meta, **evaluation_meta}

            else:
                result_meta = self.meta

        else:
            result_meta = evaluation_meta

        return EvaluatorResult(
            evaluator=self.name,
            score=evaluation_score,
            threshold=self.threshold,
            meta=result_meta,
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
    threshold: EvaluationScoreValue | None = None,
    execution_context: ScopeContext | None = None,
    meta: Meta | None = None,
) -> Callable[
    [EvaluatorDefinition[Value, Args]],
    Evaluator[Value, Args],
]: ...


def evaluator[Value, **Args](
    evaluation: EvaluatorDefinition[Value, Args] | None = None,
    /,
    *,
    name: str | None = None,
    threshold: EvaluationScoreValue | None = None,
    execution_context: ScopeContext | None = None,
    meta: Meta | None = None,
) -> (
    Callable[
        [EvaluatorDefinition[Value, Args]],
        Evaluator[Value, Args],
    ]
    | Evaluator[Value, Args]
):
    def wrap(
        definition: EvaluatorDefinition[Value, Args],
    ) -> Evaluator[Value, Args]:
        return Evaluator(
            name=name or definition.__name__,
            definition=definition,
            threshold=evaluation_score_value(threshold) if threshold is not None else None,
            execution_context=execution_context,
            meta=meta,
        )

    if evaluation:
        return wrap(evaluation)

    else:
        return wrap
