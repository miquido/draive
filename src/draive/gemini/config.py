from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = (
    "GeminiEmbeddingConfig",
    "GeminiGenerationConfig",
    "GeminiLiveConfig",
    "GeminiSafetyConfig",
    "GeminiSafetyThreshold",
)

GeminiSafetyThreshold = Literal["off", "none", "high", "medium", "low"]


class GeminiSafetyConfig(Config):
    # gemini safety is really bad and often triggers false positive so disabling by default
    harm_category_hate_speech_threshold: GeminiSafetyThreshold = "off"
    harm_category_harassment_threshold: GeminiSafetyThreshold = "off"
    harm_category_sexually_explicit_threshold: GeminiSafetyThreshold = "off"
    harm_category_civic_integrity_threshold: GeminiSafetyThreshold = "off"
    harm_category_dangerous_content_threshold: GeminiSafetyThreshold = "off"


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
    thinking_budget: int | Missing = MISSING
    safety: GeminiSafetyConfig | Missing = MISSING


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
