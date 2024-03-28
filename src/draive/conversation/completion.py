from typing import Literal, Protocol, overload, runtime_checkable

from draive.conversation.message import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationStreamingUpdate,
)
from draive.lmm import LMMCompletionStream
from draive.tools import Toolbox
from draive.types import Memory, UpdateSend

__all__ = [
    "ConversationCompletionStream",
    "ConversationCompletion",
]


ConversationCompletionStream = LMMCompletionStream


@runtime_checkable
class ConversationCompletion(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: str,
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        tools: Toolbox | None = None,
        stream: Literal[True],
    ) -> ConversationCompletionStream:
        ...

    @overload
    async def __call__(
        self,
        *,
        instruction: str,
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        tools: Toolbox | None = None,
        stream: UpdateSend[ConversationStreamingUpdate],
    ) -> ConversationMessage:
        ...

    @overload
    async def __call__(
        self,
        *,
        instruction: str,
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        tools: Toolbox | None = None,
    ) -> ConversationMessage:
        ...

    async def __call__(  # noqa: PLR0913
        self,
        *,
        instruction: str,
        input: ConversationMessage | ConversationMessageContent,  # noqa: A002
        memory: Memory[ConversationMessage] | None = None,
        tools: Toolbox | None = None,
        stream: UpdateSend[ConversationStreamingUpdate] | bool = False,
    ) -> ConversationCompletionStream | ConversationMessage:
        ...
