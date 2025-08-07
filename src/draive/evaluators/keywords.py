from collections.abc import Callable, Sequence

from draive.evaluation import EvaluationScore, evaluator
from draive.multimodal import Multimodal, MultimodalContent

__all__ = (
    "forbidden_keywords_evaluator",
    "required_keywords_evaluator",
)


@evaluator(name="required_keywords")
async def required_keywords_evaluator(
    evaluated: Multimodal,
    /,
    *,
    keywords: Sequence[str],
    require_all: bool = True,
    normalization: Callable[[str], str] | None = None,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if not keywords:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Keywords list was empty!"},
        )

    text_normalization: Callable[[str], str]
    if normalization is not None:
        text_normalization = normalization

    else:
        text_normalization = _lowercased

    normalized_text: str = text_normalization(MultimodalContent.of(evaluated).to_str())
    required: int = len(keywords)
    matching: int = len(
        [keyword for keyword in keywords if text_normalization(keyword) in normalized_text]
    )

    return EvaluationScore.of(
        matching == required if require_all else matching / required,
    )


@evaluator(name="forbidden_keywords")
async def forbidden_keywords_evaluator(
    evaluated: Multimodal,
    /,
    *,
    keywords: Sequence[str],
    require_none: bool = True,
    normalization: Callable[[str], str] | None = None,
) -> EvaluationScore:
    if not evaluated:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Input was empty!"},
        )

    if not keywords:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "Keywords list was empty!"},
        )

    text_normalization: Callable[[str], str]
    if normalization is not None:
        text_normalization = normalization

    else:
        text_normalization = _lowercased

    normalized_text: str = text_normalization(MultimodalContent.of(evaluated).to_str())
    forbidden: int = len(keywords)
    matching: int = len(
        [keyword for keyword in keywords if text_normalization(keyword) in normalized_text]
    )

    return EvaluationScore.of(
        matching == 0 if require_none else 1.0 - matching / forbidden,
    )


def _lowercased(text: str) -> str:
    return text.lower()
