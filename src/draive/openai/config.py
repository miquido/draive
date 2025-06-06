from collections.abc import Sequence
from typing import Literal, TypedDict

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = (
    "OpenAIChatConfig",
    "OpenAIEmbeddingConfig",
    "OpenAIImageGenerationConfig",
    "OpenAIModerationConfig",
    "OpenAIRealtimeConfig",
)


class AudioResponseFormat(TypedDict):
    format: Literal["wav", "mp3"]
    voice: Literal["ash", "ballad", "coral", "sage", "verse"]


class OpenAIChatConfig(Config):
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


class OpenAIRealtimeConfig(Config):
    model: str
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    input_audio_noise_reduction: Literal["near_field", "far_field"] | Missing = MISSING
    voice: str | Missing = MISSING
    vad_type: Literal["server_vad", "semantic_vad"] | Missing = MISSING
    vad_eagerness: Literal["low", "medium", "high", "auto"] = "auto"
    transcribe_model: str | Missing = MISSING


class OpenAIEmbeddingConfig(Config):
    model: str
    dimensions: int | Missing = MISSING
    batch_size: int = 128
    timeout: float | Missing = MISSING


class OpenAIImageGenerationConfig(Config):
    model: str
    result: Literal["url", "b64_json"]
    quality: Literal["standard", "hd"] = "standard"
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    style: Literal["vivid", "natural"] = "vivid"
    timeout: float | Missing = MISSING


class OpenAIModerationConfig(Config):
    model: str = "omni-moderation-latest"
    timeout: float | Missing = MISSING
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
    illicit_threshold: float | Missing = MISSING
    illicit_violent_threshold: float | Missing = MISSING
