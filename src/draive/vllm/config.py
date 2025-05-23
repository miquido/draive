from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = (
    "VLLMChatConfig",
    "VLLMEmbeddingConfig",
)


class VLLMChatConfig(Config):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    frequency_penalty: float | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    vision_details: Literal["auto", "low", "high"] | Missing = MISSING
    parallel_tool_calls: bool | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class VLLMEmbeddingConfig(Config):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
    timeout: float | Missing = MISSING
