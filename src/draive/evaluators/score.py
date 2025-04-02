from typing import Any

from draive.evaluation import EvaluationScore
from draive.parameters import DataModel, Field, ParameterValidationContext

__all__ = ("CommonScoreModel",)


def _score_validator(
    value: Any,
    context: ParameterValidationContext,
) -> float:
    match value:
        case float() as float_value:
            return float_value

        case int() as int_value:
            return float(int_value)

        case str() as str_value:
            try:
                return float(str_value)

            except Exception as exc:
                raise TypeError(f"Expected 'float', received '{type(value).__name__}'") from exc

        case _:
            raise TypeError(f"Expected 'float', received '{type(value).__name__}'")


class CommonScoreModel(DataModel):
    comment: str | None = Field(
        description="Explanation of the score",
        default=None,
    )
    score: float = Field(
        description="Decimal score value",
        validator=_score_validator,
    )

    def normalized(
        self,
        divider: float | None = None,
    ) -> EvaluationScore:
        return EvaluationScore(
            value=self.score / divider if divider else self.score,
            comment=self.comment,
        )
