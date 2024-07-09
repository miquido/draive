from pathlib import Path
from typing import Any

from draive.gemini.config import GeminiConfig
from draive.scope import ctx
from draive.sentencepiece import sentencepiece_tokenize_text
from draive.utils import cache

__all__ = [
    "gemini_tokenize_text",
]


def gemini_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return sentencepiece_tokenize_text(
        text=text,
        model_path=_model_path(ctx.state(GeminiConfig).model),
        **extra,
    )


@cache(limit=2)
def _model_path(model_name: str) -> str:
    model_file: str = _mapping.get(model_name, "gemini_tokenizer.model")
    return str(Path(__file__).parent / "tokens" / model_file)


_mapping: dict[str, str] = {
    "gemini-1.5-flash": "gemini_tokenizer.model",
    "gemini-1.5-pro": "gemini_tokenizer.model",
}
