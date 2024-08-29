from typing import Final, Literal

__all__ = [
    "EXCELLENT",
    "FAIR",
    "GOOD",
    "NONE",
    "PERFECT",
    "POOR",
    "threshold_value",
    "Threshold",
]

type Threshold = Literal["none", "poor", "fair", "good", "excellent", "perfect"] | float

NONE: Final[float] = 0.0
POOR: Final[float] = 0.1
FAIR: Final[float] = 0.3
GOOD: Final[float] = 0.5
EXCELLENT: Final[float] = 0.7
PERFECT: Final[float] = 0.9


def threshold_value(  # noqa: PLR0911
    threshold: Threshold,
    /,
) -> float:
    match threshold:
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

        case value:
            assert 0 <= value <= 1, "Threshold value has to be in range from 0 to 1"  # nosec: B101
            return value
