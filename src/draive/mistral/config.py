from collections.abc import Sequence

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = (
    "MistralChatConfig",
    "MistralEmbeddingConfig",
)


class MistralChatConfig(Config):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    seed: int | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class MistralEmbeddingConfig(Config):
    model: str
    batch_size: int = 128
