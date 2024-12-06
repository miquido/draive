from haiway import State

from draive.embedding.types import ValueEmbedder

__all__ = [
    "ImageEmbedding",
    "TextEmbedding",
]


class TextEmbedding(State):
    embed: ValueEmbedder[str]


class ImageEmbedding(State):
    embed: ValueEmbedder[bytes]
