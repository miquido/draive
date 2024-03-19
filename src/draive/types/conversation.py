from collections.abc import AsyncIterator
from typing import Literal, Protocol, Self, overload, runtime_checkable

from draive.types.memory import Memory
from draive.types.model import Model
from draive.types.progress import ProgressUpdate
from draive.types.string import StringConvertible
from draive.types.tool import ToolCallProgress
from draive.types.toolset import Toolset

__all__ = [
    "ConversationMessageTextContent",
    "ConversationMessageImageReferenceContent",
    "ConversationMessageContent",
    "ConversationMessage",
    "ConversationCompletion",
    "ConversationResponseStream",
    "ConversationStreamingPartialMessage",
    "ConversationStreamingUpdate",
]


class ConversationMessageTextContent(Model):
    text: str


class ConversationMessageImageReferenceContent(Model):
    url: str


ConversationMessageContent = (
    ConversationMessageTextContent | ConversationMessageImageReferenceContent
)


class ConversationMessage(Model):
    role: str
    author: str | None = None
    content: list[ConversationMessageContent] | str
    timestamp: str | None = None


class ConversationStreamingPartialMessage(Model):
    content: str


ConversationStreamingUpdate = ConversationStreamingPartialMessage | ToolCallProgress


class ConversationResponseStream(Protocol):
    def as_json(self) -> AsyncIterator[str]:
        ...

    def __aiter__(self) -> Self:
        ...

    async def __anext__(self) -> ConversationStreamingUpdate:
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
        stream: ProgressUpdate[ConversationStreamingUpdate],
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
        stream: ProgressUpdate[ConversationStreamingUpdate] | bool = False,
    ) -> ConversationResponseStream | ConversationMessage:
        ...
