from typing import Final, Literal

__all__ = (
    "EvaluationScoreValue",
    "evaluation_score_value",
    "evaluation_score_verifier",
)

type EvaluationScoreValue = (
    Literal[
        "none",
        "poor",
        "fair",
        "good",
        "excellent",
        "perfect",
        "max",
    ]
    | float
    | bool
)

NONE: Final[float] = 0.0
POOR: Final[float] = 0.1
FAIR: Final[float] = 0.3
GOOD: Final[float] = 0.5
EXCELLENT: Final[float] = 0.7
PERFECT: Final[float] = 0.9
MAX: Final[float] = 1.0


def evaluation_score_value(  # noqa: C901, PLR0911
    value: EvaluationScoreValue,
    /,
) -> float:
    """
    Convert an evaluation score value to a normalized float.

    Converts various score representations to a float between 0.0 and 1.0.

    Parameters
    ----------
    value : EvaluationScoreValue
        Score value to convert. Can be:
        - Named level: "none", "poor", "fair", "good", "excellent", "perfect", "max"
        - Float: Must be between 0.0 and 1.0
        - Boolean: False becomes 0.0, True becomes 1.0

    Returns
    -------
    float
        Normalized score between 0.0 and 1.0

    Raises
    ------
    AssertionError
        If float value is outside [0, 1] range
    ValueError
        If value is not a recognized score type
    """
    match value:
        case float() as value:
            assert 0 <= value <= 1, "Score value has to be in range from 0 to 1"  # nosec: B101
            return value

        case False:
            return 0.0

        case True:
            return 1.0

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

        case "max":
            return MAX

        case int() as value:
            assert 0 <= value <= 1, "Score value has to be in range from 0 to 1"  # nosec: B101
            return float(value)

        case _:
            raise ValueError(f"Invalid evaluation score value - {value}")


def evaluation_score_verifier(
    value: float,
) -> float:
    """
    Verify that a score value is within valid range.

    Parameters
    ----------
    value : float
        Float value to verify

    Raises
    ------
    ValueError
        If value is not between 0 and 1 inclusive
    """
    if 0 <= value <= 1:
        return value  # valid

    raise ValueError(f"Evaluation score has to be a value between 0 and 1, received: {value}")
