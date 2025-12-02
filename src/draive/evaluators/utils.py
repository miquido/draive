from typing import Final, cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue
from draive.multimodal import MultimodalContent, MultimodalTag

__all__ = (
    "FORMAT_INSTRUCTION",
    "extract_evaluation_result",
)

COMMENT_TAG_NAME: Final[str] = "comment"
RATING_TAG_NAME: Final[str] = "rating"

FORMAT_INSTRUCTION: Final[str] = f"""\
<FORMAT>
Respond using exactly the following XML structure and include no other text before or after it.
Include both the rating and the comment tags exactly once.

  <{COMMENT_TAG_NAME}>Concise, step-by-step justification that supports the rating. Do not leave empty.</{COMMENT_TAG_NAME}>
  <{RATING_TAG_NAME}>Single, lowercase rating name chosen from the available ratings list (no quotes or extra text).</{RATING_TAG_NAME}>
</FORMAT>
"""  # noqa: E501


def extract_evaluation_result(
    content: MultimodalContent,
    /,
) -> EvaluationScore:
    rating_tag: MultimodalTag | None = None
    comment_tag: MultimodalTag | None = None

    for tag in content.tags():
        if tag.name.lower() == RATING_TAG_NAME:
            if rating_tag is not None:
                raise ValueError(f"Invalid evaluator result - multiple rating tags:\n{content}")

            rating_tag = tag

        elif tag.name.lower() == COMMENT_TAG_NAME:
            if comment_tag is not None:
                raise ValueError(f"Invalid evaluator result - multiple comment tags:\n{content}")

            comment_tag = tag

    if not rating_tag:
        raise ValueError(f"Invalid evaluator result - missing rating:\n{content}")

    try:
        return EvaluationScore.of(
            cast(EvaluationScoreValue, rating_tag.content.to_str().strip().lower()),
            meta={
                "comment": comment_tag.content.to_str().strip() if comment_tag is not None else None
            },
        )

    except Exception as exc:
        raise ValueError(f"Invalid evaluator result - {exc}:\n{content}") from exc
