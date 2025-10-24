from typing import Final, cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue
from draive.multimodal import MultimodalContent

__all__ = (
    "FORMAT_INSTRUCTION",
    "RATING_TAG_NAME",
    "extract_evaluation_result",
)

REASONING_TAG_NAME: Final[str] = "reasoning"
RATING_TAG_NAME: Final[str] = "rating"

FORMAT_INSTRUCTION: Final[str] = """\
<FORMAT>
Respond using exactly the following XML structure:
  <REASONING>Concise, step-by-step justification that supports the rating.</REASONING>
  <RATING>Selected rating value with exactly one of the allowed rating names (no quotes or extra text)</RATING>
</FORMAT>
"""  # noqa: E501


def extract_evaluation_result(
    content: MultimodalContent,
    /,
) -> EvaluationScore:
    reasoning: str | None = None
    rating: str | None = None
    for tag in content.tags():
        name: str = tag.name.lower()
        if name == RATING_TAG_NAME:
            rating = tag.content.to_str()

        elif name == REASONING_TAG_NAME:
            reasoning = tag.content.to_str()

    if not rating:
        raise ValueError(f"Invalid evaluator result - missing rating:\n{content}")

    return EvaluationScore.of(
        cast(EvaluationScoreValue, rating.strip().lower()),
        meta={"comment": reasoning},
    )
