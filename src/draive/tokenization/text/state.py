from draive.tokenization.text.counter import TextTokenCounter
from draive.types import State

__all__ = [
    "TextTokenization",
]


class TextTokenization(State):
    count_text_tokens: TextTokenCounter
