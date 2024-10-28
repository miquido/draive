from typing import Literal, TypedDict

from haiway import MISSING, Missing, State

__all__ = [
    "MistralChatConfig",
    "MistralEmbeddingConfig",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class MistralChatConfig(State):
    model: str = "open-mistral-7b"
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    seed: int | None | Missing = MISSING
    max_tokens: int | Missing = MISSING
    response_format: ResponseFormat | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING


class MistralEmbeddingConfig(State):
    model: str = "mistral-embed"
    batch_size: int = 128
