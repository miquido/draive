from collections.abc import Sequence
from typing import Literal

from draive.evaluation import (
    EvaluationScore,
    EvaluationScoreValue,
    EvaluatorResult,
    evaluator,
)
from draive.evaluation.agreement import (
    cohen_kappa,
    quadratic_weighted_kappa,
    quantize_score,
)
from draive.evaluation.value import evaluation_score_value

__all__ = ("cohen_kappa_evaluator",)


type Rating = EvaluationScoreValue | EvaluationScore | EvaluatorResult


def _rating_value(
    rating: Rating,
    /,
) -> float:
    match rating:
        case EvaluationScore() as score:
            return score.value

        case EvaluatorResult() as result:
            return result.score

        case value:
            return evaluation_score_value(value)


@evaluator(name="cohen_kappa")
async def cohen_kappa_evaluator(
    evaluated: Sequence[Rating],
    /,
    *,
    reference: Sequence[Rating],
    weighting: Literal["nominal", "quadratic"] = "quadratic",
) -> EvaluationScore:
    """
    Evaluate inter-rater agreement between two aligned sets of ratings using Cohen's kappa.

    Typically used to compare an automatic rater's scores (``evaluated``) against human
    gold labels (``reference``). Each side accepts named levels, floats/bools, ``EvaluationScore``
    instances, or ``EvaluatorResult`` instances (its ``score`` is used) - so an evaluator's own
    outputs can be fed in directly.

    Parameters
    ----------
    evaluated : Sequence[Rating]
        Ratings from the rater under inspection (e.g. an automatic evaluator).
    reference : Sequence[Rating]
        Aligned ground-truth ratings (e.g. human labels).
    weighting : Literal["nominal", "quadratic"]
        Which kappa drives the returned score - "quadratic" (default, ordinal, penalizes
        distant disagreements more) or "nominal" (unweighted). Both variants are always
        reported in the result metadata.

    Returns
    -------
    EvaluationScore
        Score equal to the selected kappa, clamped to [0, 1] (negative agreement maps to 0.0).
        Metadata carries both kappa variants, exact agreement, sample count and weighting.
        Misaligned or empty inputs yield a failed (0.0) score with an explanatory comment.
    """
    if len(evaluated) != len(reference):
        return EvaluationScore.of(
            0.0,
            meta={
                "comment": "Rating sequences must be aligned pairs, "
                f"received lengths {len(evaluated)} vs {len(reference)}",
                "cohen_kappa": 0.0,
                "quadratic_weighted_kappa": 0.0,
                "exact_agreement": 0.0,
                "sample_count": 0,
                "weighting": weighting,
            },
        )

    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={
                "comment": "Ratings were empty!",
                "cohen_kappa": 0.0,
                "quadratic_weighted_kappa": 0.0,
                "exact_agreement": 0.0,
                "sample_count": 0,
                "weighting": weighting,
            },
        )

    evaluated_bins: list[str] = [quantize_score(_rating_value(rating)) for rating in evaluated]
    reference_bins: list[str] = [quantize_score(_rating_value(rating)) for rating in reference]

    nominal: float = cohen_kappa(reference_bins, evaluated_bins)
    quadratic: float = quadratic_weighted_kappa(reference_bins, evaluated_bins)
    exact_agreement: float = sum(
        1 for left, right in zip(reference_bins, evaluated_bins, strict=True) if left == right
    ) / len(evaluated_bins)

    selected_kappa: float = quadratic if weighting == "quadratic" else nominal

    return EvaluationScore.of(
        max(0.0, selected_kappa),
        meta={
            "cohen_kappa": nominal,
            "quadratic_weighted_kappa": quadratic,
            "exact_agreement": exact_agreement,
            "sample_count": len(evaluated_bins),
            "weighting": weighting,
        },
    )
