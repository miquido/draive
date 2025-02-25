from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Missing, State

__all__ = [
    "GeminiEmbeddingConfig",
    "GeminiGenerationConfig",
]


class GeminiGenerationConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    timeout: float | None = None
    stop_sequences: Sequence[str] | None = None
    media_resolution: Literal["low", "medium", "high"] | Missing = MISSING


class GeminiEmbeddingConfig(State):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
