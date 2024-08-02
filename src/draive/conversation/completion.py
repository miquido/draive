from collections.abc import Sequence
from typing import Any, Literal, Protocol, overload, runtime_checkable

from draive.conversation.model import ConversationMessage, ConversationResponseStream
from draive.instructions import Instruction
from draive.lmm import Toolbox
from draive.types import Memory

__all__ = [
    "ConversationCompletion",
]


@runtime_checkable
class ConversationCompletion(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        message: ConversationMessage,
        memory: Memory[Sequence[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: Literal[True],
        **extra: Any,
    ) -> ConversationResponseStream: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        message: ConversationMessage,
        memory: Memory[Sequence[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        message: ConversationMessage,
        memory: Memory[Sequence[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: bool,
        **extra: Any,
    ) -> ConversationResponseStream | ConversationMessage: ...

    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        message: ConversationMessage,
        memory: Memory[Sequence[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: bool = False,
        **extra: Any,
    ) -> ConversationResponseStream | ConversationMessage: ...
