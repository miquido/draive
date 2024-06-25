from typing import Any

from anthropic import Anthropic
from tokenizers import Tokenizer

from draive.utils import cache

__all__ = [
    "anthropic_tokenize_text",
]


def anthropic_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return _tokenizer().encode(text).ids  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


@cache(limit=1)
def _tokenizer() -> Tokenizer:
    return Anthropic().get_tokenizer()
