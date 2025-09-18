from collections.abc import Mapping, Sequence

from haiway import MISSING, Configuration, Missing

__all__ = (
    "MistralChatConfig",
    "MistralEmbeddingConfig",
    "MistralModerationConfig",
)


class MistralChatConfig(Configuration):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    seed: int | Missing = MISSING
    max_output_tokens: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class MistralEmbeddingConfig(Configuration):
    model: str
    batch_size: int = 128


class MistralModerationConfig(Configuration):
    model: str = "mistral-moderation-latest"
    category_thresholds: Mapping[str, float] | Missing = MISSING
