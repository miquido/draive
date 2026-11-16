from collections.abc import Sequence
from typing import Literal

from haiway import Configuration, Missing

__all__ = (
    "OllamaChatConfig",
    "OllamaEmbeddingConfig",
)


class OllamaChatConfig(Configuration):
    model: str
    temperature: float | Missing = Missing
    top_k: int | Missing = Missing
    top_p: float | Missing = Missing
    seed: int | Missing | None = Missing
    max_output_tokens: int | Missing = Missing
    stop_sequences: Sequence[str] | Missing = Missing
    # reasoning models require explicitly requesting thinking to report it
    thinking: bool | Literal["low", "medium", "high"] | Missing = Missing


class OllamaEmbeddingConfig(Configuration):
    model: str
    concurrent: bool = False
    batch_size: int = 32
