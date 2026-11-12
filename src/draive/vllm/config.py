from collections.abc import Sequence
from typing import Literal

from haiway import Configuration, Missing

__all__ = (
    "VLLMChatConfig",
    "VLLMEmbeddingConfig",
)


class VLLMChatConfig(Configuration):
    model: str
    temperature: float | Missing = Missing
    top_p: float | Missing = Missing
    frequency_penalty: float | Missing = Missing
    max_output_tokens: int | Missing = Missing
    seed: int | Missing = Missing
    vision_details: Literal["auto", "low", "high"] | Missing = Missing
    parallel_tool_calls: bool | Missing = Missing
    stop_sequences: Sequence[str] | Missing = Missing


class VLLMEmbeddingConfig(Configuration):
    model: str
    dimensions: int | Missing = Missing
    batch_size: int = 32
