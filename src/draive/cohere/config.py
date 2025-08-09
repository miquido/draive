from typing import Literal

from haiway import Configuration

__all__ = (
    "CohereImageEmbeddingConfig",
    "CohereTextEmbeddingConfig",
)


class CohereTextEmbeddingConfig(Configuration):
    model: str
    purpose: Literal[
        "search_query",
        "search_document",
        "clustering",
        "classification",
    ]
    batch_size: int = 128


class CohereImageEmbeddingConfig(Configuration):
    model: str
    batch_size: int = 16
