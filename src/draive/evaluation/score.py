from typing import Any, Self

from draive.evaluation.value import EvaluationScoreValue, evaluation_score_value
from draive.parameters import DataModel, Field

__all__ = [
    "EvaluationScore",
]


def _verifier(
    value: float,
) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"Score has to have a value between 0 and 1, received: {value}")


class EvaluationScore(DataModel):
    @classmethod
    def of(
        cls,
        score: EvaluationScoreValue,
        /,
        *,
        comment: str | None = None,
    ) -> Self:
        return cls(
            value=evaluation_score_value(score),
            comment=comment,
        )

    value: float = Field(
        description="Score value, between 0 (failure) and 1 (success)",
        verifier=_verifier,
    )
    comment: str | None = Field(
        description="Explanation of the score",
        default=None,
    )

    def __eq__(self, other: Any) -> bool:
        match other:
            case float() as value:
                return self.value == value

            case EvaluationScore() as score:
                return self.value == score.value

            case _:
                return NotImplemented

    def __ne__(self, other: Any) -> bool:
        match other:
            case float() as value:
                return self.value != value

            case EvaluationScore() as score:
                return self.value != score.value

            case _:
                return NotImplemented

    def __lt__(self, other: Any) -> bool:
        match other:
            case float() as value:
                return self.value < value

            case EvaluationScore() as score:
                return self.value < score.value

            case _:
                return NotImplemented

    def __le__(self, other: Any) -> bool:
        match other:
            case float() as value:
                return self.value <= value

            case EvaluationScore() as score:
                return self.value <= score.value

            case _:
                return NotImplemented

    def __gt__(self, other: Any) -> bool:
        match other:
            case float() as value:
                return self.value > value

            case EvaluationScore() as score:
                return self.value > score.value

            case _:
                return NotImplemented

    def __ge__(self, other: Any) -> bool:
        match other:
            case float() as value:
                return self.value >= value

            case EvaluationScore() as score:
                return self.value >= score.value

            case _:
                return NotImplemented
