from collections.abc import Mapping, Sequence
from typing import Literal

from haiway import Configuration, Missing

__all__ = (
    "MistralChatConfig",
    "MistralEmbeddingConfig",
    "MistralModerationConfig",
)


class MistralChatConfig(Configuration):
    model: str
    temperature: float | Missing = Missing
    top_p: float | Missing = Missing
    seed: int | Missing = Missing
    max_output_tokens: int | Missing = Missing
    stop_sequences: Sequence[str] | Missing = Missing
    # reasoning models require explicitly requesting the reasoning prompt
    # mode to produce and report their thinking
    prompt_mode: Literal["reasoning"] | Missing | None = Missing


class MistralEmbeddingConfig(Configuration):
    model: str
    batch_size: int = 128


class MistralModerationConfig(Configuration):
    model: str = "mistral-moderation-latest"
    category_thresholds: Mapping[str, float] | Missing = Missing
