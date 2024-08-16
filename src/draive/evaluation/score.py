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
    value: float = Field(
        description="Score value, between 0 (failure) and 1 (success)",
        verifier=_verifier,
    )
    comment: str | None = Field(
        description="Explanation of the score",
        default=None,
    )
