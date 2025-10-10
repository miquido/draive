from typing import Annotated, Any, Self

from haiway import META_EMPTY, Description, Meta, MetaValues, Validator

from draive.evaluation.value import (
    EvaluationScoreValue,
    evaluation_score_value,
    evaluation_score_verifier,
)
from draive.parameters import DataModel

__all__ = ("EvaluationScore",)


class EvaluationScore(DataModel):
    """
    Evaluation score with value and optional comment.

    Represents a normalized evaluation score between 0 and 1, with an
    optional comment explaining the score. Supports comparison operations
    with other scores or float values.

    Attributes
    ----------
    value : float
        Score value between 0 (failure) and 1 (success)
    comment : str | None
        Optional explanation of the score
    """

    @classmethod
    def of(
        cls,
        score: EvaluationScoreValue,
        /,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Create an EvaluationScore from a score value.

        Parameters
        ----------
        score : EvaluationScoreValue
            Score value as float, bool, or named level
        meta: Meta | MetaValues | None = None
            Metadata about the score

        Returns
        -------
        Self
            New EvaluationScore instance
        """
        return cls(
            value=evaluation_score_value(score),
            meta=Meta.of(meta),
        )

    value: Annotated[
        float,
        Description("Score value, between 0 (failure) and 1 (success)"),
        Validator(evaluation_score_verifier),
    ]
    meta: Annotated[
        Meta,
        Description("Metadata about the score"),
    ] = META_EMPTY

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another score or float.

        Parameters
        ----------
        other : Any
            Value to compare with (float or EvaluationScore)

        Returns
        -------
        bool
            True if values are equal
        """
        if isinstance(other, float):
            return self.value == other

        elif isinstance(other, EvaluationScore):
            return self.value == other.value

        else:
            return False

    def __ne__(self, other: Any) -> bool:
        """
        Check inequality with another score or float.

        Parameters
        ----------
        other : Any
            Value to compare with (float or EvaluationScore)

        Returns
        -------
        bool
            True if values are not equal
        """
        if isinstance(other, float):
            return self.value != other

        elif isinstance(other, EvaluationScore):
            return self.value != other.value

        else:
            return False

    def __lt__(self, other: Any) -> bool:
        """
        Check if less than another score or float.

        Parameters
        ----------
        other : Any
            Value to compare with (float or EvaluationScore)

        Returns
        -------
        bool
            True if this value is less than other
        """
        if isinstance(other, float):
            return self.value < other

        elif isinstance(other, EvaluationScore):
            return self.value < other.value

        else:
            return False

    def __le__(self, other: Any) -> bool:
        """
        Check if less than or equal to another score or float.

        Parameters
        ----------
        other : Any
            Value to compare with (float or EvaluationScore)

        Returns
        -------
        bool
            True if this value is less than or equal to other
        """
        if isinstance(other, float):
            return self.value <= other

        elif isinstance(other, EvaluationScore):
            return self.value <= other.value

        else:
            return False

    def __gt__(self, other: Any) -> bool:
        """
        Check if greater than another score or float.

        Parameters
        ----------
        other : Any
            Value to compare with (float or EvaluationScore)

        Returns
        -------
        bool
            True if this value is greater than other
        """
        if isinstance(other, float):
            return self.value > other

        elif isinstance(other, EvaluationScore):
            return self.value > other.value

        else:
            return False

    def __ge__(self, other: Any) -> bool:
        """
        Check if greater than or equal to another score or float.

        Parameters
        ----------
        other : Any
            Value to compare with (float or EvaluationScore)

        Returns
        -------
        bool
            True if this value is greater than or equal to other
        """
        if isinstance(other, float):
            return self.value >= other

        elif isinstance(other, EvaluationScore):
            return self.value >= other.value

        else:
            return False

    def __hash__(self) -> int:
        """
        Get hash value for the score.

        Returns
        -------
        int
            Hash of the value and metadata
        """
        return hash((self.value, self.meta))
