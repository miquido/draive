from typing import Any

from anthropic import Anthropic
from haiway import cache
from tokenizers import Tokenizer

from draive.tokenization import Tokenization

__all__ = [
    "anthropic_tokenizer",
]


def anthropic_tokenizer() -> Tokenization:
    def anthropic_tokenize_text(
        text: str,
        **extra: Any,
    ) -> list[int]:
        return _tokenizer().encode(text).ids  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    return Tokenization(tokenize_text=anthropic_tokenize_text)


@cache(limit=1)
def _tokenizer() -> Tokenizer:
    return Anthropic().get_tokenizer()
