from collections.abc import Sequence
from typing import Literal, TypedDict

from haiway import MISSING, Missing, State

__all__ = [
    "OpenAIChatConfig",
    "OpenAIEmbeddingConfig",
    "OpenAIImageGenerationConfig",
    "OpenAIModerationConfig",
    "OpenAISystemFingerprint",
]


class AudioResponseFormat(TypedDict):
    format: Literal["wav", "mp3"]
    voice: Literal["ash", "ballad", "coral", "sage", "verse"]


class OpenAIChatConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    frequency_penalty: float | Missing = MISSING
    max_tokens: int | Missing = MISSING
    seed: int | None | Missing = MISSING
    audio_response_format: AudioResponseFormat | Missing = MISSING
    vision_details: Literal["auto", "low", "high"] | Missing = MISSING
    parallel_tool_calls: bool | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class OpenAIEmbeddingConfig(State):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
    encoding_format: Literal["float", "base64"] | Missing = MISSING
    timeout: float | Missing = MISSING


class OpenAIImageGenerationConfig(State):
    model: str
    quality: Literal["standard", "hd"] = "standard"
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    style: Literal["vivid", "natural"] = "vivid"
    timeout: float | Missing = MISSING
    response_format: Literal["url", "b64_json"] = "b64_json"


class OpenAISystemFingerprint(State):
    system_fingerprint: str


class OpenAIModerationConfig(State):
    harassment_threshold: float | Missing = MISSING
    harassment_threatening_threshold: float | Missing = MISSING
    hate_threshold: float | Missing = MISSING
    hate_threatening_threshold: float | Missing = MISSING
    self_harm_threshold: float | Missing = MISSING
    self_harm_instructions_threshold: float | Missing = MISSING
    self_harm_intent_threshold: float | Missing = MISSING
    sexual_threshold: float | Missing = MISSING
    sexual_minors_threshold: float | Missing = MISSING
    violence_threshold: float | Missing = MISSING
    violence_graphic_threshold: float | Missing = MISSING
