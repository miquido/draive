from typing import Final, cast

from draive.evaluation import EvaluationScore, EvaluationScoreValue
from draive.models import (
    ModelContext,
    ModelOutput,
    ModelToolRequest,
)
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTag

__all__ = (
    "FORMAT_INSTRUCTION",
    "extract_evaluation_result",
    "is_empty_content",
    "model_context_multimodal",
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


def is_empty_content(
    content: Multimodal,
    /,
) -> bool:
    """
    Check whether multimodal content has no assessable payload.

    Parameters
    ----------
    content : Multimodal
        Content to inspect.

    Returns
    -------
    bool
        True when content has no parts or only whitespace-only text. Resource
        and artifact parts are considered assessable payloads.
    """
    multimodal_content: MultimodalContent = MultimodalContent.of(content)

    if not multimodal_content:
        return True

    if multimodal_content.contains_resources or multimodal_content.contains_artifacts:
        return False

    return not multimodal_content.to_str().strip()


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


def model_context_multimodal(
    context: ModelContext,
    /,
) -> MultimodalContent:
    parts: list[str | MultimodalContent] = []
    for index, element in enumerate(context, start=1):
        if isinstance(element, ModelOutput):
            parts.append(f"\n<CONTEXT_ELEMENT index='{index}' role='model'>")
            for block in element.output:
                if isinstance(block, MultimodalContent):
                    parts.append(block)
                elif isinstance(block, ModelToolRequest):
                    parts.append(f"<TOOL_CALL tool='{block.tool}'/>")
                # ModelReasoning intentionally skipped
            parts.append("</CONTEXT_ELEMENT>")
        else:  # ModelInput
            parts.append(f"\n<CONTEXT_ELEMENT index='{index}' role='user'>")
            for block in element.input:
                if isinstance(block, MultimodalContent):
                    parts.append(block)
                else:
                    parts.append(f"<TOOL_RESPONSE tool='{block.tool}' status='{block.status}'>")
                    parts.append(block.content)
                    parts.append("</TOOL_RESPONSE>")
            parts.append("</CONTEXT_ELEMENT>")
    return MultimodalContent.of(*parts)
