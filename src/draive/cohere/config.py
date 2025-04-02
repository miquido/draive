from typing import Literal

from haiway import State

__all__ = (
    "CohereImageEmbeddingConfig",
    "CohereTextEmbeddingConfig",
)


class CohereTextEmbeddingConfig(State):
    model: str
    purpose: Literal[
        "search_query",
        "search_document",
        "clustering",
        "classification",
    ]
    batch_size: int = 128


class CohereImageEmbeddingConfig(State):
    model: str
    batch_size: int = 16
