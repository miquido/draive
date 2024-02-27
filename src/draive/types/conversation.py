from collections.abc import AsyncIterator
from typing import Literal, Protocol, Self, overload

from draive.types.memory import Memory
from draive.types.message import ConversationMessage
from draive.types.model import Model
from draive.types.string import StringConvertible
from draive.types.toolset import Toolset

__all__ = [
    "ConversationCompletion",
    "ConversationResponseStream",
    "ConversationStreamingActionStatus",
    "ConversationStreamingAction",
    "ConversationStreamingPart",
]


ConversationStreamingActionStatus = Literal["STARTED", "PROGRESS", "FINISHED", "FAILED"]


class ConversationStreamingAction(Model):
    id: str
    action: Literal["TOOL_CALL"]
    name: str
    status: ConversationStreamingActionStatus


class ConversationStreamingPart(Model):
    actions: list[ConversationStreamingAction]
    message: ConversationMessage


class ConversationResponseStream(Protocol):
    def as_json(self) -> AsyncIterator[str]:
        ...

    def __aiter__(self) -> Self:
        ...

    async def __anext__(self) -> ConversationStreamingPart:
        ...


class ConversationCompletion(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: str,
        input: ConversationMessage | StringConvertible,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        toolset: Toolset | None = None,
        stream: Literal[True],
    ) -> ConversationResponseStream:
        ...

    @overload
    async def __call__(
        self,
        *,
        instruction: str,
        input: ConversationMessage | StringConvertible,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        toolset: Toolset | None = None,
    ) -> ConversationMessage:
        ...

    async def __call__(  # noqa: PLR0913
        self,
        *,
        instruction: str,
        input: ConversationMessage | StringConvertible,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        toolset: Toolset | None = None,
        stream: bool = False,
    ) -> ConversationResponseStream | ConversationMessage:
        ...
