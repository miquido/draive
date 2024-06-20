from pathlib import Path
from typing import Any, cast

from sentencepiece import SentencePieceProcessor  # pyright: ignore[reportMissingTypeStubs]

from draive.gemini.config import GeminiConfig
from draive.scope import ctx
from draive.utils import cache

__all__ = [
    "gemini_tokenize_text",
]


def gemini_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return cast(
        list[int],
        _encoding(model_name=ctx.state(GeminiConfig).model).encode(text),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    )


@cache(limit=4)
def _encoding(model_name: str) -> SentencePieceProcessor:
    model_file: str = _mapping.get(model_name, "gemini_tokenizer.model")
    return SentencePieceProcessor(model_file=str(Path(__file__).parent / "tokens" / model_file))  # pyright: ignore[reportCallIssue]


_mapping: dict[str, str] = {
    "gemini-1.5-flash": "gemini_tokenizer.model",
    "gemini-1.5-pro": "gemini_tokenizer.model",
}
