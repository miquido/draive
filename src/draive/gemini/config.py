from typing import Literal

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "GeminiConfig",
    "GeminiEmbeddingConfig",
]


class GeminiConfig(DataModel):
    model: (
        Literal[
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        | str
    ) = "gemini-1.5-flash"
    temperature: float = 0.0
    response_format: Literal["text/plain", "application/json"] | Missing = MISSING
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING


class GeminiEmbeddingConfig(DataModel):
    model: Literal["text-embedding-004",] | str = "text-embedding-004"
