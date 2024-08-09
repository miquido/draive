from typing import Literal, TypedDict

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "OpenAIChatConfig",
    "OpenAIEmbeddingConfig",
    "OpenAIImageGenerationConfig",
    "OpenAIModerationConfig",
    "OpenAISystemFingerprint",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class OpenAIChatConfig(DataModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.75
    top_p: float | Missing = MISSING
    frequency_penalty: float | Missing = MISSING
    max_tokens: int = 2048
    seed: int | None | Missing = MISSING
    response_format: ResponseFormat | Missing = MISSING
    vision_details: Literal["auto", "low", "high"] | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING


class OpenAIEmbeddingConfig(DataModel):
    model: str = "text-embedding-3-small"
    dimensions: int | Missing = MISSING
    batch_size: int = 128
    encoding_format: Literal["float", "base64"] | Missing = MISSING
    timeout: float | Missing = MISSING


class OpenAIImageGenerationConfig(DataModel):
    model: str = "dall-e-2"
    quality: Literal["standard", "hd"] = "standard"
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    style: Literal["vivid", "natural"] = "vivid"
    timeout: float | Missing = MISSING
    response_format: Literal["url", "b64_json"] = "b64_json"


class OpenAISystemFingerprint(DataModel):
    system_fingerprint: str


class OpenAIModerationConfig(DataModel):
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
