from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, overload, runtime_checkable

from draive.conversation.types import (
    ConversationMemory,
    ConversationMessage,
    ConversationStreamElement,
)
from draive.instructions import Instruction
from draive.tools import Toolbox

__all__ = ("ConversationCompleting",)


@runtime_checkable
class ConversationCompleting(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: ConversationMessage,
        memory: ConversationMemory,
        toolbox: Toolbox,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: ConversationMessage,
        memory: ConversationMemory,
        toolbox: Toolbox,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[ConversationStreamElement]: ...

    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: ConversationMessage,  # noqa: A002
        memory: ConversationMemory,
        toolbox: Toolbox,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[ConversationStreamElement] | ConversationMessage: ...
