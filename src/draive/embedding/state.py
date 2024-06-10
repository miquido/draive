from draive.embedding.embedder import Embedder
from draive.parameters import State

__all__ = [
    "TextEmbedding",
    "ImageEmbedding",
]


class TextEmbedding(State):
    embed: Embedder[str]


class ImageEmbedding(State):
    embed: Embedder[bytes]
