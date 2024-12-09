from collections.abc import Sequence

from haiway import MISSING, Missing, State

__all__ = [
    "GeminiConfig",
    "GeminiEmbeddingConfig",
]


class GeminiConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class GeminiEmbeddingConfig(State):
    model: str
    batch_size: int = 128
