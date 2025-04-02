from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Missing, State

__all__ = (
    "GeminiEmbeddingConfig",
    "GeminiGenerationConfig",
    "GeminiLiveConfig",
)


class GeminiGenerationConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
    speech_voice_name: str | Missing = MISSING
    media_resolution: Literal["low", "medium", "high"] | Missing = MISSING


class GeminiLiveConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
    speech_voice_name: str | Missing = MISSING
    media_resolution: Literal["low", "medium", "high"] | Missing = MISSING


class GeminiEmbeddingConfig(State):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
