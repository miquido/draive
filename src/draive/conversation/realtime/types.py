from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from draive.conversation.types import ConversationStreamElement, RealtimeConversationMemory
from draive.instructions import Instruction
from draive.tools import Toolbox

__all__ = ("RealtimeConversationStarting",)


@runtime_checkable
class RealtimeConversationStarting(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input_stream: AsyncIterator[ConversationStreamElement],
        memory: RealtimeConversationMemory,
        toolbox: Toolbox,
        **extra: Any,
    ) -> AsyncIterator[ConversationStreamElement]: ...
