from draive.embedding.embedder import Embedder
from draive.types import State

__all__ = [
    "Embedding",
]


class Embedding(State):
    embed_text: Embedder[str]
