from draive.scope import ctx
from draive.tokenization.state import Tokenization

__all__ = [
    "count_text_tokens",
]


def tokenize_text(
    text: str,
) -> list[int]:
    return ctx.state(Tokenization).tokenize_text(text=text)


def count_text_tokens(
    text: str,
) -> int:
    return len(ctx.state(Tokenization).tokenize_text(text=text))
