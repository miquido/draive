from draive.openai import openai_count_text_tokens
from draive.scope import ScopeState
from draive.tokenization.token_counter import TextTokenCounter

__all__ = [
    "Tokenization",
]


class Tokenization(ScopeState):
    count_text_tokens: TextTokenCounter = openai_count_text_tokens
