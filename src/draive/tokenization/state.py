from draive.parameters import State
from draive.tokenization.text import TextTokenizer

__all__ = [
    "Tokenization",
]


class Tokenization(State):
    tokenize_text: TextTokenizer
