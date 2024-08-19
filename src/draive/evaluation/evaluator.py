from collections.abc import Callable
from typing import Protocol, Self, cast, final, overload, runtime_checkable

from draive.evaluation.score import EvaluationScore
from draive.parameters import DataModel, Field, ParameterPath
from draive.scope import ctx
from draive.utils import freeze

__all__ = [
    "evaluator",
    "Evaluator",
    "EvaluatorResult",
    "PreparedEvaluator",
    "EvaluatorDefinition",
]


class EvaluatorResult(DataModel):
    evaluator: str = Field(
        description="Name of evaluator",
    )
    score: EvaluationScore = Field(
        description="Evaluation score",
    )
    threshold: float = Field(
        description="Score threshold required to pass evaluation",
    )
    meta: dict[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
    )

    @property
    def passed(self) -> bool:
        return self.score.value >= self.threshold


class EvaluationResult(DataModel):
    @classmethod
    async def of(
        cls,
        score: EvaluationScore | float | bool,
        /,
        meta: dict[str, str | float | int | bool | None] | None = None,
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
            meta=meta,
        )

    score: EvaluationScore = Field(
        description="Evaluation score",
    )
    meta: dict[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
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
    def __init__(
        self,
        name: str,
        definition: EvaluatorDefinition[Value, Args],
        threshold: float | None,
    ) -> None:
        assert (  # nosec: B101
            threshold is None or 0 <= threshold <= 1
        ), "Evaluation threshold has to be between 0 and 1"

        self._definition: EvaluatorDefinition[Value, Args] = definition
        self.name: str = name
        self.threshold: float = threshold or 1

        freeze(self)

    def with_threshold(
        self,
        threshold: float,
    ) -> Self:
        return self.__class__(
            name=self.name,
            definition=self._definition,
            threshold=threshold,
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
        mapping: Callable[[Mapped], Value] | ParameterPath[Mapped, Value] | Value,
        /,
    ) -> "Evaluator[Mapped, Args]":
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
        )

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorResult:
        evaluation_score: EvaluationScore
        evaluation_meta: dict[str, str | float | int | bool | None] | None
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
                    evaluation_meta = None

                case float() as score_value:
                    evaluation_score = EvaluationScore(value=score_value)
                    evaluation_meta = None

                case passed:
                    evaluation_score = EvaluationScore(value=1 if passed else 0)
                    evaluation_meta = None

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

        return EvaluatorResult(
            evaluator=self.name,
            score=evaluation_score,
            threshold=self.threshold,
            meta=evaluation_meta,
        )


@overload
def evaluator[Value, **Args](
    definition: EvaluatorDefinition[Value, Args] | None = None,
    /,
) -> Evaluator[Value, Args]: ...


@overload
def evaluator[Value, **Args](
    *,
    name: str | None = None,
    threshold: float | None = None,
) -> Callable[
    [EvaluatorDefinition[Value, Args]],
    Evaluator[Value, Args],
]: ...


def evaluator[Value, **Args](
    evaluation: EvaluatorDefinition[Value, Args] | None = None,
    *,
    name: str | None = None,
    threshold: float | None = None,
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
            threshold=threshold,
        )

    if evaluation:
        return wrap(evaluation)

    else:
        return wrap
