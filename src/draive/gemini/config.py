from typing import Literal

from haiway import MISSING, Missing, State

__all__ = [
    "GeminiConfig",
    "GeminiEmbeddingConfig",
]


class GeminiConfig(State):
    model: str = "gemini-1.5-flash"
    temperature: float = 1.0
    response_format: Literal["text/plain", "application/json"] | Missing = MISSING
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING


class GeminiEmbeddingConfig(State):
    model: str = "embedding-gecko-001"
    batch_size: int = 128
