from draive.tokenization.text import TextTokenizer
from draive.types import State

__all__ = [
    "Tokenization",
]


class Tokenization(State):
    tokenize_text: TextTokenizer
