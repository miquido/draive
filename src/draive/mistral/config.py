from typing import Literal, TypedDict

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "MistralChatConfig",
    "MistralEmbeddingConfig",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class MistralChatConfig(DataModel):
    model: str = "open-mistral-7b"
    temperature: float = 0.75
    top_p: float | Missing = MISSING
    seed: int | None | Missing = MISSING
    max_tokens: int = 2048
    response_format: ResponseFormat | Missing = MISSING
    timeout: float | Missing = MISSING


class MistralEmbeddingConfig(DataModel):
    model: str = "mistral-embed"
    batch_size: int = 128
