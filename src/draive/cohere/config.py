from typing import Literal

from draive.configuration import Config

__all__ = (
    "CohereImageEmbeddingConfig",
    "CohereTextEmbeddingConfig",
)


class CohereTextEmbeddingConfig(Config):
    model: str
    purpose: Literal[
        "search_query",
        "search_document",
        "clustering",
        "classification",
    ]
    batch_size: int = 128


class CohereImageEmbeddingConfig(Config):
    model: str
    batch_size: int = 16
