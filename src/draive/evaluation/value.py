from typing import Final, Literal

__all__ = [
    "EXCELLENT",
    "FAIR",
    "GOOD",
    "NONE",
    "PERFECT",
    "POOR",
    "evaluation_score_value",
    "EvaluationScoreValue",
]

type EvaluationScoreValue = Literal["none", "poor", "fair", "good", "excellent", "perfect"] | float

NONE: Final[float] = 0.0
POOR: Final[float] = 0.1
FAIR: Final[float] = 0.3
GOOD: Final[float] = 0.5
EXCELLENT: Final[float] = 0.7
PERFECT: Final[float] = 0.9


def evaluation_score_value(  # noqa: PLR0911
    value: EvaluationScoreValue,
    /,
) -> float:
    match value:
        case "none":
            return NONE

        case "poor":
            return POOR

        case "fair":
            return FAIR

        case "good":
            return GOOD

        case "excellent":
            return EXCELLENT

        case "perfect":
            return PERFECT

        case float() as value:
            assert 0 <= value <= 1, "Threshold value has to be in range from 0 to 1"  # nosec: B101
            return value

        case _:
            raise ValueError("Invalid evaluation score value - %s", value)
