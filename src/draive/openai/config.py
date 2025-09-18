from typing import Literal, TypedDict

from haiway import MISSING, Configuration, Missing
from openai.types.realtime.realtime_audio_config_input_param import RealtimeAudioConfigInputParam
from openai.types.realtime.realtime_audio_config_output_param import RealtimeAudioConfigOutputParam

__all__ = (
    "OpenAIEmbeddingConfig",
    "OpenAIImageGenerationConfig",
    "OpenAIModerationConfig",
    "OpenAIRealtimeConfig",
    "OpenAIResponsesConfig",
)


class AudioResponseFormat(TypedDict):
    format: Literal["wav", "mp3"]
    voice: Literal["ash", "ballad", "coral", "sage", "verse"]


class OpenAIResponsesConfig(Configuration):
    model: Literal["gpt-5", "gpt-5-mini", "gpt-5-nano"] | str
    temperature: float | Missing = MISSING
    vision_details: Literal["auto", "low", "high"] = "auto"
    verbosity: Literal["low", "medium", "high"] | Missing = MISSING
    reasoning: Literal["minimal", "low", "medium", "high"] | Missing = MISSING
    reasoning_summary: Literal["auto", "concise", "detailed"] = "auto"
    parallel_tool_calls: bool = True
    truncation: Literal["auto", "disabled"] = "auto"
    max_output_tokens: int | None = None
    safety_identifier: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"


class ServerVADParameters(TypedDict):
    vad_type: Literal["server_vad"]
    threshold: float
    silence_duration_ms: int
    prefix_padding_ms: int


class OpenAIRealtimeConfig(Configuration):
    model: Literal["gpt-realtime"] | str = "gpt-realtime"
    input_parameters: RealtimeAudioConfigInputParam
    output_parameters: RealtimeAudioConfigOutputParam


class OpenAIEmbeddingConfig(Configuration):
    model: Literal["text-embedding-3-large", "text-embedding-3-small"] | str = (
        "text-embedding-3-small"
    )
    dimensions: int | Missing = MISSING
    batch_size: int = 128


class OpenAIImageGenerationConfig(Configuration):
    model: Literal["gpt-image-1"] | str = "gpt-image-1"
    result: Literal["url", "b64_json"]
    quality: Literal["standard", "hd"] = "standard"
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    style: Literal["vivid", "natural"] = "vivid"


class OpenAIModerationConfig(Configuration):
    model: Literal["omni-moderation-latest"] | str = "omni-moderation-latest"
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
