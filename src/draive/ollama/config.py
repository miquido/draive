from collections.abc import Sequence

from haiway import MISSING, Missing, State

__all__ = ["OllamaChatConfig", "OllamaEmbeddingConfig"]


class OllamaChatConfig(State):
    model: str
    temperature: float = 1.0
    top_k: int | Missing = MISSING
    top_p: float | Missing = MISSING
    seed: int | None | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class OllamaEmbeddingConfig(State):
    model: str
    concurrent: bool = False
    batch_size: int = 32
