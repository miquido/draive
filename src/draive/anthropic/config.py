from collections.abc import Sequence

from haiway import MISSING, Configuration, Missing

__all__ = ("AnthropicConfig",)


class AnthropicConfig(Configuration):
    model: str
    temperature: float | Missing = MISSING
    max_output_tokens: int = 2048
    thinking_budget: int | None | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
