from collections.abc import AsyncIterator
from typing import Literal, Protocol, Self, overload, runtime_checkable

from draive.types.memory import Memory
from draive.types.message import ConversationMessage
from draive.types.model import Model
from draive.types.streaming import StreamingProgressUpdate
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
    data: Model | None = None


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


@runtime_checkable
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
        stream: StreamingProgressUpdate[ConversationStreamingPart],
    ) -> ConversationMessage:
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
        stream: StreamingProgressUpdate[ConversationStreamingPart] | bool = False,
    ) -> ConversationResponseStream | ConversationMessage:
        ...
