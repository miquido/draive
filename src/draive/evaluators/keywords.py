from collections.abc import Callable, Sequence

from draive.evaluation import EvaluationScore, evaluator
from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("keywords_evaluator",)


@evaluator(name="keywords")
async def keywords_evaluator(
    content: Multimodal,
    /,
    *,
    keywords: Sequence[str],
    normalization: Callable[[str], str] | None = None,
) -> EvaluationScore:
    if not content:
        return EvaluationScore(
            value=0,
            comment="Input was empty!",
        )

    if not keywords:
        return EvaluationScore(
            value=0,
            comment="Keywords list was empty!",
        )

    text_normalization: Callable[[str], str]
    if normalization is not None:
        text_normalization = normalization

    else:
        text_normalization = _lowercased

    normalized_text: str = text_normalization(MultimodalContent.of(content).to_str())
    return EvaluationScore(
        value=len(
            [keyword for keyword in keywords if text_normalization(keyword) in normalized_text]
        )
        / len(keywords),
    )


def _lowercased(text: str) -> str:
    return text.lower()
