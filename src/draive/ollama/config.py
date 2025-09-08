from collections.abc import Sequence

from haiway import MISSING, Configuration, Missing

__all__ = (
    "OllamaChatConfig",
    "OllamaEmbeddingConfig",
)


class OllamaChatConfig(Configuration):
    model: str
    temperature: float = 1.0
    top_k: int | Missing = MISSING
    top_p: float | Missing = MISSING
    seed: int | None | Missing = MISSING
    max_output_tokens: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class OllamaEmbeddingConfig(Configuration):
    model: str
    concurrent: bool = False
    batch_size: int = 32
