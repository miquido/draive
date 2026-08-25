from collections.abc import Sequence
from typing import Literal

from haiway import MISSING, Configuration, Missing

__all__ = ("AnthropicConfig",)


class AnthropicConfig(Configuration):
    model: Literal["claude-opus-5", "claude-sonnet-5", "claude-fable-5"] | str
    # thinking consumes this budget before any content is produced,
    # a small limit makes the model terminate without answering
    max_output_tokens: int = 16384
    # thinking runs in adaptive mode unless explicitly disabled, which not every
    # model allows; its content is never exposed, only its signature is reported
    thinking: Literal["adaptive", "disabled"] | Missing = MISSING
    # depth of both thinking and the answer itself
    effort: Literal["low", "medium", "high", "xhigh", "max"] | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING
