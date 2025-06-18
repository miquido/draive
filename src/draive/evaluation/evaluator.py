from asyncio import gather
from collections.abc import Callable
from typing import Any, Protocol, Self, cast, final, overload, runtime_checkable

from haiway import AttributePath, ScopeContext, ctx

from draive.commons import META_EMPTY, Meta, MetaValues
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
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            evaluator=evaluator,
            score=EvaluationScore.of(
                score,
                comment=score_comment,
            ),
            threshold=evaluation_score_value(threshold),
            meta=Meta.of(meta),
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
        status: str = "passed" if self.passed else "failed"
        if include_details:
            comment: str = f"'{self.score.comment}'" if self.score.comment else "N/A"

            return (
                f"<evaluator name='{self.evaluator}' status='{status}'>"
                f"\n<score>{self.score.value}</score>"
                f"\n<threshold>{self.threshold}</threshold>"
                f"\n<relative_score>{self.relative_score * 100:.2f}%</relative_score>"
                f"\n<comment>{comment}</comment>"
                "\n</evaluator>"
            )

        else:
            return (
                f"{self.evaluator}: {status}, comment: {self.score.comment}"
                if self.score.comment
                else f"{self.evaluator}: {status}"
            )

    @property
    def relative_score(self) -> float:
        if self.threshold <= 0:
            return 1.0

        return min(1.0, self.score.value / self.threshold)

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

    def __hash__(self) -> int:  # explicitly using super to silence warnings
        return hash((self.evaluator, self.score, self.threshold))


class EvaluationResult(DataModel):
    @classmethod
    async def of(
        cls,
        score: EvaluationScore | float | bool,
        /,
        meta: Meta | MetaValues | None = None,
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
            meta=Meta.of(meta),
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

    __slots__ = (
        "_definition",
        "_execution_context",
        "meta",
        "name",
        "threshold",
    )

    def __init__(
        self,
        name: str,
        definition: EvaluatorDefinition[Value, Args],
        threshold: float | None,
        execution_context: ScopeContext | None,
        meta: Meta,
    ) -> None:
        assert (  # nosec: B101
            threshold is None or 0 <= threshold <= 1
        ), "Evaluation threshold has to be between 0 and 1"
        assert "\n" not in name  # nosec: B101

        self._definition: EvaluatorDefinition[Value, Args]
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
        self.name: str
        object.__setattr__(
            self,
            "name",
            name.lower().replace(" ", "_"),
        )
        self.threshold: float
        object.__setattr__(
            self,
            "threshold",
            1 if threshold is None else threshold,
        )
        self.meta: Meta
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
        meta: Meta | MetaValues,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            threshold=self.threshold,
            execution_context=self._execution_context,
            meta=self.meta.merged_with(meta),
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
            evaluation_meta = Meta({"exception": str(exc)})

        result_meta: Meta
        if self.meta:
            if evaluation_meta:
                result_meta = self.meta.merged_with(evaluation_meta)

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
    threshold: EvaluationScoreValue | None = None,
    execution_context: ScopeContext | None = None,
    meta: Meta | MetaValues | None = None,
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
            meta=Meta.of(meta),
        )

    if evaluation:
        return wrap(evaluation)

    else:
        return wrap
