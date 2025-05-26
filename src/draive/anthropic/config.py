from collections.abc import Sequence

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = ("AnthropicConfig",)


class AnthropicConfig(Config):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    max_tokens: int = 2048
    thinking_tokens_budget: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
