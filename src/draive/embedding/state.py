from draive.openai import openai_embed_text
from draive.scope import ScopeState
from draive.types import Embedder

__all__ = [
    "Embedding",
]


class Embedding(ScopeState):
    embed_text: Embedder = openai_embed_text
