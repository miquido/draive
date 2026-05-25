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
    ]
    | float
    | bool
)

NONE: Final[float] = 0.0
POOR: Final[float] = 0.2
FAIR: Final[float] = 0.4
GOOD: Final[float] = 0.6
EXCELLENT: Final[float] = 0.8
PERFECT: Final[float] = 1.0


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
        - Named level: "none", "poor", "fair", "good", "excellent", "perfect"
        - Float: Must be between 0.0 and 1.0
        - Boolean: False becomes 0.0, True becomes 1.0

    Returns
    -------
    float
        Normalized score between 0.0 and 1.0

    Raises
    ------
    ValueError
        If a numeric value is outside [0, 1] range, or the value is not a
        recognized score type
    """
    match value:
        # Match bool before int/float — bool is a subclass of int (not float),
        # but listing it first makes the intent explicit and order-independent.
        case False:
            return 0.0

        case True:
            return 1.0

        case float() as value:
            return evaluation_score_verifier(value)

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

        case int() as value:
            return evaluation_score_verifier(float(value))

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
