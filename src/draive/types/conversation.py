from collections.abc import AsyncIterator
from typing import Literal, Protocol, Self, overload, runtime_checkable

from draive.types.images import ImageContent
from draive.types.memory import Memory
from draive.types.model import Model
from draive.types.multimodal import MultimodalContent
from draive.types.progress import ProgressUpdate
from draive.types.tool import ToolCallProgress
from draive.types.toolset import Toolset

__all__ = [
    "ConversationMessageContent",
    "ConversationMessage",
    "ConversationCompletion",
    "ConversationResponseStream",
    "ConversationStreamingPartialMessage",
    "ConversationStreamingUpdate",
]

ConversationMessageContent = MultimodalContent


class ConversationMessage(Model):
    role: str
    author: str | None = None
    content: ConversationMessageContent
    timestamp: str | None = None

    @property
    def has_media(self) -> bool:
        if isinstance(self.content, str):
            return False
        elif isinstance(self.content, ImageContent):
            return True
        else:
            return any(not isinstance(element, str) for element in self.content)

    @property
    def content_string(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, ImageContent):
            return ""
        else:
            return "\n".join(element for element in self.content if isinstance(element, str))


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
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
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
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
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
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        toolset: Toolset | None = None,
    ) -> ConversationMessage:
        ...

    async def __call__(  # noqa: PLR0913
        self,
        *,
        instruction: str,
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        toolset: Toolset | None = None,
        stream: ProgressUpdate[ConversationStreamingUpdate] | bool = False,
    ) -> ConversationResponseStream | ConversationMessage:
        ...
