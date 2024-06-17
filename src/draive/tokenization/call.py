from typing import Any

from draive.scope import ctx
from draive.tokenization.state import Tokenization

__all__ = [
    "tokenize_text",
    "count_text_tokens",
]


def tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return ctx.state(Tokenization).tokenize_text(
        text=text,
        **extra,
    )


def count_text_tokens(
    text: str,
    **extra: Any,
) -> int:
    return len(
        ctx.state(Tokenization).tokenize_text(
            text=text,
            **extra,
        )
    )
