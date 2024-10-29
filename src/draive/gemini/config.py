from typing import Literal

from haiway import MISSING, Missing

from draive.parameters import DataModel

__all__ = [
    "GeminiConfig",
    "GeminiEmbeddingConfig",
]


class GeminiConfig(DataModel):
    model: str = "gemini-1.5-flash"
    temperature: float = 0.75
    response_format: Literal["text/plain", "application/json"] | Missing = MISSING
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int = 2048
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING


class GeminiEmbeddingConfig(DataModel):
    model: str = "embedding-gecko-001"
    batch_size: int = 128
