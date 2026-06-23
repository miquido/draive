from typing import Annotated, Literal, Self

from haiway import Description, State, Validator

from draive.evaluation.value import (
    EvaluationScoreValue,
    evaluation_score_value,
    evaluation_score_verifier,
)

__all__ = (
    "EvaluationReference",
    "reference_conformance",
)


class EvaluationReference(State, serializable=True):
    """
    Accepted score window for reference-based evaluation.

    Expresses the ground-truth as an inclusive range ``[lower, upper]`` in the
    normalized score space rather than a single point, so a prediction that lands
    anywhere inside the window is treated as full agreement. This is the interface
    for describing acceptable alterations around an expected rating.

    Attributes
    ----------
    lower : float
        Inclusive lower bound of the accepted window, between 0 and 1.
    upper : float
        Inclusive upper bound of the accepted window, between 0 and 1.
    """

    @classmethod
    def of(
        cls,
        target: EvaluationScoreValue,
        /,
        *,
        tolerance: float = 0.0,
    ) -> Self:
        """
        Create a reference centered on a target value with a symmetric tolerance.

        Parameters
        ----------
        target : EvaluationScoreValue
            Expected score as a named level, float, or bool.
        tolerance : float
            Accepted deviation on each side of ``target`` as a fraction of the target
            value (a percentage where 1.0 == 100%), by default 0.0 (an exact single-point
            reference). For example a target of 0.6 with ``tolerance=0.2`` accepts ±0.12,
            yielding the window [0.48, 0.72]. Must be non-negative; the window is clamped
            to [0, 1].

        Returns
        -------
        Self
            Reference window ``[target - target * tolerance, target + target * tolerance]``.
        """
        assert 0.0 <= tolerance <= 1.0  # nosec: B101

        center: float = evaluation_score_value(target)
        margin: float = center * tolerance
        return cls(
            lower=max(0.0, center - margin),
            upper=min(1.0, center + margin),
        )

    @classmethod
    def between(
        cls,
        lower: EvaluationScoreValue,
        upper: EvaluationScoreValue,
        /,
    ) -> Self:
        """
        Create a reference from an explicit accepted range.

        Parameters
        ----------
        lower : EvaluationScoreValue
            Inclusive lower bound as a named level, float, or bool.
        upper : EvaluationScoreValue
            Inclusive upper bound as a named level, float, or bool.

        Returns
        -------
        Self
            Reference window ``[lower, upper]``.

        Raises
        ------
        ValueError
            If ``lower`` is greater than ``upper``.
        """
        low: float = evaluation_score_value(lower)
        high: float = evaluation_score_value(upper)
        assert low > high  # nosec: B101

        return cls(
            lower=low,
            upper=high,
        )

    lower: Annotated[
        float,
        Description("Inclusive lower bound of the accepted score window"),
        Validator(evaluation_score_verifier),
    ]
    upper: Annotated[
        float,
        Description("Inclusive upper bound of the accepted score window"),
        Validator(evaluation_score_verifier),
    ]

    def contains(
        self,
        score: float,
        /,
    ) -> bool:
        """
        Check whether a score falls within the accepted window (inclusive).

        Parameters
        ----------
        score : float
            Predicted score to test against the window bounds.

        Returns
        -------
        bool
            ``True`` if ``score`` lies within ``[lower, upper]`` inclusive,
            otherwise ``False``.
        """
        return self.lower <= score <= self.upper


def reference_conformance(
    score: float,
    reference: EvaluationReference,
    /,
    *,
    weighting: Literal["quadratic", "nominal"] = "quadratic",
) -> float:
    """
    Score how well a predicted value conforms to a reference window.

    A prediction inside the window is full agreement (1.0). Outside, the result
    depends on the weighting: ``nominal`` rejects outright (0.0), while ``quadratic``
    awards partial credit that decays with the squared distance to the nearest
    window boundary.

    Parameters
    ----------
    score : float
        Predicted score in [0, 1].
    reference : EvaluationReference
        Accepted score window.
    weighting : Literal["quadratic", "nominal"]
        Falloff applied to predictions outside the window, by default "quadratic".

    Returns
    -------
    float
        Conformance score in [0, 1]; 1.0 when within the window.
    """
    assert weighting in ("quadratic", "nominal"), (  # nosec: B101
        f"Unsupported weighting '{weighting}'; expected 'quadratic' or 'nominal'"
    )

    if reference.contains(score):
        return 1.0

    if weighting == "nominal":
        return 0.0

    # Quadratic partial credit on the miss, measured from the nearest boundary.
    # The gap lies in [0, 1], so a fixed unit scale keeps the falloff independent
    # of the window's width and placement.
    gap: float = reference.lower - score if score < reference.lower else score - reference.upper
    return max(0.0, 1.0 - gap * gap)
