from draive.scope import ctx
from draive.tokenization.text.state import TextTokenization

__all__ = [
    "count_text_tokens",
]


def count_text_tokens(
    text: str,
) -> int:
    return ctx.state(TextTokenization).count_text_tokens(text=text)
