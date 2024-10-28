from typing import Any

from anthropic import Anthropic
from haiway import cache
from tokenizers import Tokenizer

from draive.tokenization import TextTokenizing

__all__ = [
    "anthropic_text_tokenizer",
]


def anthropic_text_tokenizer() -> TextTokenizing:
    def anthropic_tokenize_text(
        text: str,
        **extra: Any,
    ) -> list[int]:
        return _tokenizer().encode(text).ids  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    return anthropic_tokenize_text


@cache(limit=1)
def _tokenizer() -> Tokenizer:
    return Anthropic().get_tokenizer()
