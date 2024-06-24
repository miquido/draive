from typing import Literal, TypedDict

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "AnthropicConfig",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class AnthropicConfig(DataModel):
    model: str = "claude-3-5-sonnet-20240620"
    temperature: float = 0.0
    top_p: float | Missing = MISSING
    max_tokens: int = 2048
    timeout: float | Missing = MISSING
