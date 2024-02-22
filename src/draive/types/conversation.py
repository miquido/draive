from typing import Protocol

from draive.types.memory import Memory
from draive.types.message import ConversationMessage
from draive.types.string import StringConvertible
from draive.types.toolset import Toolset

__all__ = [
    "ConversationCompletion",
]


class ConversationCompletion(Protocol):
    async def __call__(
        self,
        *,
        instruction: str,
        input: ConversationMessage | StringConvertible,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        toolset: Toolset | None = None,
    ) -> ConversationMessage:
        ...
