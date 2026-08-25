from typing import Literal

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


class OpenAIResponsesConfig(Configuration):
    model: (
        Literal[
            "gpt-5.6-terra",
            "gpt-5.6-sol",
            "gpt-5.6-luna",
            "gpt-5.5",
        ]
        | str
    )
    vision_details: Literal["auto", "low", "high"] = "auto"
    verbosity: Literal["low", "medium", "high"] | Missing = MISSING
    # "max" requires the gpt-5.6 generation, older models stop at "xhigh"
    reasoning: Literal["none", "low", "medium", "high", "xhigh", "max"] | Missing = MISSING
    reasoning_summary: Literal["auto", "concise", "detailed"] = "auto"
    # scope of the reasoning carried into a turn, the default differs between model
    # generations - gpt-5.6 uses "all_turns" where gpt-5.5 uses "current_turn"
    reasoning_context: Literal["auto", "current_turn", "all_turns"] | Missing = MISSING
    # "pro" requires an effort of at least "medium" on the models providing it
    reasoning_mode: Literal["standard", "pro"] | Missing = MISSING
    truncation: Literal["auto", "disabled"] = "auto"
    max_output_tokens: int | Missing = MISSING
    safety_identifier: str | Missing = MISSING
    service_tier: Literal["auto", "default", "flex", "priority", "fast"] = "auto"
    # extended caching is opt-in and not offered by every model, the default
    # retention applies when this is left out
    prompt_cache_retention: Literal["in_memory", "24h"] | Missing = MISSING


class OpenAIRealtimeConfig(Configuration):
    model: (
        Literal[
            "gpt-realtime-2.1",
            "gpt-realtime-2.1-mini",
            "gpt-realtime-2",
            "gpt-realtime-1.5",
            "gpt-realtime",
            "gpt-realtime-mini",
        ]
        | str
    ) = "gpt-realtime-2.1"
    input_parameters: RealtimeAudioConfigInputParam
    output_parameters: RealtimeAudioConfigOutputParam


class OpenAIEmbeddingConfig(Configuration):
    model: Literal["text-embedding-3-large", "text-embedding-3-small"] | str = (
        "text-embedding-3-small"
    )
    dimensions: int | Missing = MISSING
    batch_size: int = 128


class OpenAIImageGenerationConfig(Configuration):
    # gpt-image models always return base64 encoded content in the requested output format,
    # `response_format` and `style` are not available for them
    model: Literal["gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-2"] | str = (
        "gpt-image-1"
    )
    quality: Literal["auto", "low", "medium", "high"] = "auto"
    size: Literal["auto", "1024x1024", "1536x1024", "1024x1536"] = "auto"
    background: Literal["auto", "transparent", "opaque"] = "auto"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    output_compression: int | Missing = MISSING
    moderation: Literal["auto", "low"] = "auto"


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
