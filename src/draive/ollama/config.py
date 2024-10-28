from typing import Literal

from haiway import MISSING, Missing, State

__all__ = [
    "OllamaChatConfig",
]


class OllamaChatConfig(State):
    model: str = "llama3:8b"
    temperature: float = 1.0
    top_k: float | Missing = MISSING
    top_p: float | Missing = MISSING
    seed: int | None | Missing = MISSING
    max_tokens: int | Missing = MISSING
    response_format: Literal["text", "json"] | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: list[str] | Missing = MISSING
