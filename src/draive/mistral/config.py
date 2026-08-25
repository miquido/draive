from collections.abc import Mapping, Sequence
from typing import Literal

from haiway import MISSING, Configuration, Missing

__all__ = (
    "MistralChatConfig",
    "MistralEmbeddingConfig",
    "MistralModerationConfig",
)


class MistralChatConfig(Configuration):
    model: str
    temperature: float | Missing = MISSING
    top_p: float | Missing = MISSING
    seed: int | Missing = MISSING
    max_output_tokens: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
    # reasoning models require explicitly requesting the reasoning prompt
    # mode to produce and report their thinking
    prompt_mode: Literal["reasoning"] | Missing | None = MISSING


class MistralEmbeddingConfig(Configuration):
    model: str
    batch_size: int = 128


class MistralModerationConfig(Configuration):
    model: str = "mistral-moderation-latest"
    category_thresholds: Mapping[str, float] | Missing = MISSING
