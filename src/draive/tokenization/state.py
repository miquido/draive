from haiway import State

from draive.tokenization.types import TextTokenizing

__all__ = [
    "Tokenization",
]


class Tokenization(State):
    tokenize_text: TextTokenizing
