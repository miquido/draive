from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = (
    "GeminiEmbeddingConfig",
    "GeminiGenerationConfig",
    "GeminiLiveConfig",
)


class GeminiGenerationConfig(Config):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
    speech_voice_name: str | Missing = MISSING
    media_resolution: Literal["low", "medium", "high"] | Missing = MISSING


class GeminiLiveConfig(Config):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    top_k: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
    speech_voice_name: str | Missing = MISSING
    media_resolution: Literal["low", "medium", "high"] | Missing = MISSING


class GeminiEmbeddingConfig(Config):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
