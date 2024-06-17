from typing import Literal

from draive.parameters import DataModel
from draive.utils import MISSING, Missing

__all__ = [
    "OllamaChatConfig",
]


class OllamaChatConfig(DataModel):
    model: str = "llama3:8b"
    temperature: float = 0.0
    top_k: float | Missing = MISSING
    top_p: float | Missing = MISSING
    seed: int | None | Missing = MISSING
    max_tokens: int | Missing = MISSING
    response_format: Literal["text", "json"] | Missing = MISSING
    timeout: float | Missing = MISSING
