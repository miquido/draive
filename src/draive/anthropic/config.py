from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Configuration, Missing

__all__ = ("AnthropicConfig",)


class AnthropicConfig(Configuration):
    model: (
        Literal[
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
        ]
        | str
    )
    temperature: float = 1.0
    max_output_tokens: int = 2048
    thinking_budget: int | None | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
