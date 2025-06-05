from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, cast, final

from haiway import State, ctx

from draive.conversation.realtime.default import realtime_conversation
from draive.conversation.realtime.types import (
    ConversationMessageChunk,
    RealtimeConversationMemory,
    RealtimeConversationStarting,
)
from draive.conversation.types import (
    ConversationElement,
    ConversationStreamElement,
)
from draive.helpers import ConstantMemory
from draive.instructions import Instruction
from draive.tools import Tool, Toolbox
from draive.utils import Memory

__all__ = ("RealtimeConversation",)


@final
class RealtimeConversation(State):
    @classmethod
    async def start(
        cls,
        *,
        instruction: Instruction | None = None,
        input_stream: AsyncIterator[ConversationMessageChunk],
        memory: RealtimeConversationMemory | Iterable[Sequence[ConversationElement]] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        **extra: Any,
    ) -> AsyncIterator[ConversationStreamElement]:
        conversation: RealtimeConversation = ctx.state(cls)

        # prepare memory
        conversation_memory: RealtimeConversationMemory
        match memory if memory is not None else conversation.memory:
            case None:
                conversation_memory = ConstantMemory(recalled=())

            case Memory() as memory:
                conversation_memory = memory

            case memory_messages:
                conversation_memory = cast(
                    RealtimeConversationMemory,
                    ConstantMemory(recalled=tuple(memory_messages)),
                )

        return await conversation.starting(
            instruction=Instruction.of(instruction),
            input_stream=input_stream,
            memory=conversation_memory,
            toolbox=Toolbox.of(tools),
        )

    starting: RealtimeConversationStarting = realtime_conversation
    memory: RealtimeConversationMemory | None = None
