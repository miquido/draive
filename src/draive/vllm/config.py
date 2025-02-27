from collections.abc import Sequence
from typing import Literal, TypedDict

from haiway import MISSING, Missing, State

__all__ = [
    "VLLMChatConfig",
    "VLLMEmbeddingConfig",
]


class AudioResponseFormat(TypedDict):
    format: Literal["wav", "mp3"]
    voice: Literal["ash", "ballad", "coral", "sage", "verse"]


class VLLMChatConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    frequency_penalty: float | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | Missing = MISSING
    audio_response_format: AudioResponseFormat | Missing = MISSING
    vision_details: Literal["auto", "low", "high"] | Missing = MISSING
    parallel_tool_calls: bool | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class VLLMEmbeddingConfig(State):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
    timeout: float | Missing = MISSING
