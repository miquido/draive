from collections.abc import AsyncGenerator
from typing import Any, Literal, Protocol, overload, runtime_checkable

from draive.conversation.types import (
    ConversationMessage,
    ConversationOutputChunk,
)
from draive.models import ModelMemory, ResolveableInstructions, Toolbox

__all__ = ("ConversationCompleting",)


@runtime_checkable
class ConversationCompleting(Protocol):
    """Provider interface for a single-turn conversation completion.

    Implementations may produce a full response message or stream output chunks
    depending on the ``stream`` flag.
    """

    @overload
    async def __call__(
        self,
        *,
        instructions: ResolveableInstructions,
        toolbox: Toolbox,
        memory: ModelMemory,
        input: ConversationMessage,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    async def __call__(
        self,
        *,
        instructions: ResolveableInstructions,
        toolbox: Toolbox,
        memory: ModelMemory,
        input: ConversationMessage,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncGenerator[ConversationOutputChunk]: ...

    async def __call__(
        self,
        *,
        instructions: ResolveableInstructions,
        toolbox: Toolbox,
        memory: ModelMemory,
        input: ConversationMessage,  # noqa: A002
        stream: bool = False,
        **extra: Any,
    ) -> AsyncGenerator[ConversationOutputChunk] | ConversationMessage: ...
