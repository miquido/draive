from collections.abc import Sequence
from typing import Literal

from haiway import Configuration, Missing

__all__ = ("AnthropicConfig",)


class AnthropicConfig(Configuration):
    model: (
        Literal[
            "claude-fable-5-2",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        ]
        | str
    )
    # thinking consumes this budget before any content is produced,
    # a small limit makes the model terminate without answering
    max_output_tokens: int = 16384
    # thinking runs in adaptive mode unless requested otherwise, models differ in what they
    # accept - Opus 5.5 and Fable 5.2 accept only adaptive, Sonnet 5.5 rejects disabled and
    # offers between_tools; its content is never exposed, only its signature is reported
    thinking: Literal["adaptive", "between_tools", "disabled"] | Missing = Missing
    # depth of both thinking and the answer itself
    effort: Literal["low", "medium", "high", "xhigh", "max"] | Missing = Missing
    stop_sequences: Sequence[str] | Missing = Missing
