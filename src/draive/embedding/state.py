from draive.embedding.embedder import ValueEmbedder
from draive.parameters import State

__all__ = [
    "TextEmbedding",
    "ImageEmbedding",
]


class TextEmbedding(State):
    embed: ValueEmbedder[str]


class ImageEmbedding(State):
    embed: ValueEmbedder[bytes]
