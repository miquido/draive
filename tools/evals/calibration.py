from collections.abc import Sequence
from typing import Literal

from haiway import State

__all__ = (
    "Calibration",
    "apply_calibration",
    "calibrate_score",
    "fit_isotonic",
    "fit_offset",
)


type CalibrationKind = Literal["offset", "isotonic"]


class Calibration(State):
    evaluator: str
    kind: CalibrationKind
    # For "offset": single bias added to raw scores.
    offset: float = 0.0
    # For "isotonic": parallel arrays of raw evaluator scores (x, sorted asc)
    # and the human scores they should map to (y, monotone non-decreasing).
    raw_points: Sequence[float] = ()
    calibrated_points: Sequence[float] = ()


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _validate_pair(
    human_scores: Sequence[float],
    evaluator_scores: Sequence[float],
    /,
) -> None:
    if len(human_scores) != len(evaluator_scores):
        raise ValueError(
            f"Score sequences differ in length: {len(human_scores)} vs {len(evaluator_scores)}"
        )

    if not human_scores:
        raise ValueError("Cannot fit calibration on empty sequences")


def fit_offset(
    evaluator: str,
    /,
    *,
    human_scores: Sequence[float],
    evaluator_scores: Sequence[float],
) -> Calibration:
    _validate_pair(human_scores, evaluator_scores)

    offset: float = sum(
        human - evaluator for human, evaluator in zip(human_scores, evaluator_scores, strict=True)
    ) / len(human_scores)

    return Calibration(evaluator=evaluator, kind="offset", offset=offset)


def fit_isotonic(
    evaluator: str,
    /,
    *,
    human_scores: Sequence[float],
    evaluator_scores: Sequence[float],
) -> Calibration:
    """
    Fit a non-decreasing piecewise-constant mapping from evaluator scores to human scores.

    Uses the pool-adjacent-violators algorithm. Requires at least one sample.
    """
    _validate_pair(human_scores, evaluator_scores)

    # Sort by evaluator score; ties keep their relative order (stable sort).
    pairs: list[tuple[float, float]] = sorted(
        zip(evaluator_scores, human_scores, strict=True), key=lambda pair: pair[0]
    )
    xs: list[float] = [x for x, _ in pairs]
    ys: list[float] = [y for _, y in pairs]

    # Pool-adjacent-violators on ys, preserving xs as anchors.
    # Each "block" is a contiguous range [start, end] with a pooled mean.
    block_start: list[int] = list(range(len(ys)))
    block_end: list[int] = list(range(len(ys)))
    block_sum: list[float] = list(ys)
    block_count: list[int] = [1] * len(ys)

    i: int = 0
    while i + 1 < len(block_sum):
        left_mean: float = block_sum[i] / block_count[i]
        right_mean: float = block_sum[i + 1] / block_count[i + 1]
        if left_mean > right_mean:
            # Pool blocks i and i+1 into one.
            block_sum[i] += block_sum[i + 1]
            block_count[i] += block_count[i + 1]
            block_end[i] = block_end[i + 1]
            del block_sum[i + 1]
            del block_count[i + 1]
            del block_start[i + 1]
            del block_end[i + 1]
            if i > 0:
                i -= 1  # may need to pool with the previous block now

        else:
            i += 1

    # Expand pooled means back to a per-point calibrated array, then collapse
    # to one (x, y) per unique x to keep the stored mapping compact.
    calibrated: list[float] = [0.0] * len(ys)
    for start, end, total, count in zip(
        block_start, block_end, block_sum, block_count, strict=True
    ):
        mean: float = total / count
        for k in range(start, end + 1):
            calibrated[k] = mean

    # Exact-equality on floats is safe here because xs are passed through
    # unchanged from the caller's evaluator_scores — no arithmetic is performed
    # on them. If a future change pre-normalises xs, switch to math.isclose.
    unique_xs: list[float] = []
    unique_ys: list[float] = []
    for x, y in zip(xs, calibrated, strict=True):
        if unique_xs and unique_xs[-1] == x:
            unique_ys[-1] = y  # last write wins; ys are monotone so equivalent
        else:
            unique_xs.append(x)
            unique_ys.append(y)

    return Calibration(
        evaluator=evaluator,
        kind="isotonic",
        raw_points=tuple(unique_xs),
        calibrated_points=tuple(unique_ys),
    )


def apply_calibration(
    calibration: Calibration,
    score: float,
    /,
) -> float:
    if calibration.kind == "offset":
        return _clamp(score + calibration.offset)

    # isotonic: piecewise-linear interpolation between learned anchors,
    # flat extrapolation past either end.
    xs: Sequence[float] = calibration.raw_points
    ys: Sequence[float] = calibration.calibrated_points
    if not xs:
        return _clamp(score)

    if score <= xs[0]:
        return _clamp(ys[0])

    if score >= xs[-1]:
        return _clamp(ys[-1])

    # Binary-search the interval [xs[i], xs[i+1]] containing score.
    low: int = 0
    high: int = len(xs) - 1
    while high - low > 1:
        mid: int = (low + high) // 2
        if xs[mid] <= score:
            low = mid

        else:
            high = mid

    x0: float = xs[low]
    x1: float = xs[high]
    y0: float = ys[low]
    y1: float = ys[high]
    if x1 == x0:
        return _clamp(y0)

    return _clamp(y0 + (y1 - y0) * (score - x0) / (x1 - x0))


def calibrate_score(
    score: float,
    /,
    *,
    calibration: Calibration | None,
) -> float:
    if calibration is None:
        return score

    return apply_calibration(calibration, score)
