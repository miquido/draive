from collections.abc import Sequence
from typing import Literal

from haiway import Configuration, Missing

__all__ = (
    "GeminiConfig",
    "GeminiEmbeddingConfig",
    "GeminiSafetyConfig",
    "GeminiSafetyThreshold",
)

GeminiSafetyThreshold = Literal[
    "off",
    "block_none",
    "block_only_high",
    "block_medium_and_above",
    "block_low_and_above",
]


class GeminiSafetyConfig(Configuration):
    # gemini safety is really bad and often triggers false positive so disabling by default
    harm_category_hate_speech_threshold: GeminiSafetyThreshold = "off"
    harm_category_harassment_threshold: GeminiSafetyThreshold = "off"
    harm_category_sexually_explicit_threshold: GeminiSafetyThreshold = "off"
    harm_category_civic_integrity_threshold: GeminiSafetyThreshold = "off"
    harm_category_dangerous_content_threshold: GeminiSafetyThreshold = "off"


class GeminiConfig(Configuration):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = Missing
    top_k: int | Missing = Missing
    max_output_tokens: int | Missing = Missing
    seed: int | Missing = Missing
    stop_sequences: Sequence[str] | Missing = Missing
    speech_voice_name: str | Missing = Missing
    speech_language_code: str | Missing = Missing
    media_resolution: Literal["low", "medium", "high"] | Missing = Missing
    thinking_budget: int | Missing = Missing
    thinking_level: Literal["minimal", "low", "medium", "high"] | Missing = Missing
    context_window_compression: bool | Missing = Missing
    # live sessions only - disable server side voice activity detection to take
    # over turn segmentation through explicit ModelSessionEvent signals
    automatic_activity_detection: bool | Missing = Missing
    safety: GeminiSafetyConfig | Missing = Missing


class GeminiEmbeddingConfig(Configuration):
    model: str
    dimensions: int | Missing = Missing
    batch_size: int = 128
