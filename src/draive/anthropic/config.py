from typing import Literal, TypedDict

from haiway import MISSING, Missing, State

__all__ = [
    "AnthropicConfig",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class AnthropicConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    max_tokens: int = 2048
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING
