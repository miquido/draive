from collections.abc import Sequence
from typing import Any, Literal, Protocol, overload, runtime_checkable

from draive.conversation.model import ConversationMessage, ConversationResponseStream
from draive.lmm import AnyTool, Toolbox
from draive.types import (
    Instruction,
    Memory,
    MultimodalContent,
    MultimodalContentConvertible,
)

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
        input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        memory: Memory[Sequence[ConversationMessage], ConversationMessage]
        | Sequence[ConversationMessage]
        | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> ConversationResponseStream: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        memory: Memory[Sequence[ConversationMessage], ConversationMessage]
        | Sequence[ConversationMessage]
        | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        memory: Memory[Sequence[ConversationMessage], ConversationMessage]
        | Sequence[ConversationMessage]
        | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
        stream: bool,
        **extra: Any,
    ) -> ConversationResponseStream | ConversationMessage: ...

    async def __call__(  # noqa: PLR0913
        self,
        *,
        instruction: Instruction | str,
        input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        memory: Memory[Sequence[ConversationMessage], ConversationMessage]
        | Sequence[ConversationMessage]
        | None = None,
        tools: Toolbox | Sequence[AnyTool] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ConversationResponseStream | ConversationMessage: ...
