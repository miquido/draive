from collections.abc import Callable
from typing import Protocol, Self, cast, final, overload, runtime_checkable

from draive.evaluation.score import Evaluation, EvaluationScore
from draive.parameters import DataModel, Field, ParameterPath
from draive.utils import freeze

__all__ = [
    "evaluator",
    "Evaluator",
    "EvaluatorResult",
    "PreparedEvaluator",
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

    @property
    def passed(self) -> bool:
        return self.score.value >= self.threshold


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
        evaluation: Evaluation[Value, Args],
        threshold: float | None,
    ) -> None:
        assert (  # nosec: B101
            threshold is None or 0 <= threshold <= 1
        ), "Evaluation threshold has to be between 0 and 1"

        self._evaluation: Evaluation[Value, Args] = evaluation
        self.name: str = name
        self.threshold: float = threshold or 1

        freeze(self)

    def with_threshold(
        self,
        threshold: float,
    ) -> Self:
        return self.__class__(
            name=self.name,
            evaluation=self._evaluation,
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
        ) -> EvaluationScore | float | bool:
            return await self._evaluation(
                mapper(value),
                *args,
                **kwargs,
            )

        return Evaluator[Mapped, Args](
            name=self.name,
            evaluation=evaluation,
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
        match await self._evaluation(
            value,
            *args,
            **kwargs,
        ):
            case float() as score_value:
                evaluation_score = EvaluationScore(value=score_value)

            case bool() as score_bool:
                evaluation_score = EvaluationScore(value=1 if score_bool else 0)

            case EvaluationScore() as score:
                evaluation_score = score

            # for whatever reason pyright wants int to be handled...
            case int() as score_int:
                evaluation_score = EvaluationScore(value=float(score_int))

        return EvaluatorResult(
            evaluator=self.name,
            score=evaluation_score,
            threshold=self.threshold,
        )


@overload
def evaluator[Value, **Args](
    evaluation: Evaluation[Value, Args] | None = None,
    /,
) -> Evaluator[Value, Args]: ...


@overload
def evaluator[Value, **Args](
    *,
    name: str | None = None,
    threshold: float | None = None,
) -> Callable[
    [Evaluation[Value, Args]],
    Evaluator[Value, Args],
]: ...


def evaluator[Value, **Args](
    evaluation: Evaluation[Value, Args] | None = None,
    *,
    name: str | None = None,
    threshold: float | None = None,
) -> (
    Callable[
        [Evaluation[Value, Args]],
        Evaluator[Value, Args],
    ]
    | Evaluator[Value, Args]
):
    def wrap(
        evaluation: Evaluation[Value, Args],
    ) -> Evaluator[Value, Args]:
        return Evaluator(
            name=name or evaluation.__name__,
            evaluation=evaluation,
            threshold=threshold,
        )

    if evaluation:
        return wrap(evaluation)

    else:
        return wrap
