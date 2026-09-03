from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Configuration, Missing

__all__ = (
    "OllamaChatConfig",
    "OllamaEmbeddingConfig",
)


class OllamaChatConfig(Configuration):
    model: str
    temperature: float | Missing = MISSING
    top_k: int | Missing = MISSING
    top_p: float | Missing = MISSING
    seed: int | Missing | None = MISSING
    max_output_tokens: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
    # reasoning models require explicitly requesting thinking to report it
    thinking: bool | Literal["low", "medium", "high"] | Missing = MISSING


class OllamaEmbeddingConfig(Configuration):
    model: str
    concurrent: bool = False
    batch_size: int = 32
