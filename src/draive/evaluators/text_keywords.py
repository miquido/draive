from collections.abc import Callable, Sequence

from draive.evaluation import EvaluationScore, evaluator

__all__ = [
    "text_keywords_evaluator",
]


@evaluator(name="text_keywords")
async def text_keywords_evaluator(
    text: str,
    /,
    keywords: Sequence[str],
    normalization: Callable[[str], str] | None = None,
) -> EvaluationScore:
    if not text:
        return EvaluationScore(
            value=0,
            comment="Input text was empty!",
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

    normalized_text: str = text_normalization(text)
    return EvaluationScore(
        value=len(
            [keyword for keyword in keywords if text_normalization(keyword) in normalized_text]
        )
        / len(keywords),
    )


def _lowercased(text: str) -> str:
    return text.lower()
