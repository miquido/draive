from typing import Protocol, runtime_checkable

from draive.parameters import DataModel, Field

__all__ = [
    "EvaluationScore",
    "Evaluation",
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


@runtime_checkable
class Evaluation[Value, **Args](Protocol):
    @property
    def __name__(self) -> str: ...

    async def __call__(
        self,
        value: Value,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluationScore | float | bool: ...
