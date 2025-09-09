from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Configuration, Missing

__all__ = (
    "VLLMChatConfig",
    "VLLMEmbeddingConfig",
)


class VLLMChatConfig(Configuration):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    frequency_penalty: float | Missing = MISSING
    max_output_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    vision_details: Literal["auto", "low", "high"] | Missing = MISSING
    parallel_tool_calls: bool | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class VLLMEmbeddingConfig(Configuration):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
