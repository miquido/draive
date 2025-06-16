from collections.abc import Iterable
from typing import Any, cast, final

from haiway import State, ctx

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.realtime.types import (
    RealtimeConversationPreparing,
    RealtimeConversationSessionScope,
)
from draive.conversation.types import ConversationMemory, ConversationMessage
from draive.helpers import ConstantMemory
from draive.instructions import Instruction
from draive.tools import Tool, Toolbox
from draive.utils import Memory

__all__ = ("RealtimeConversation",)


@final
class RealtimeConversation(State):
    @classmethod
    async def prepare(
        cls,
        *,
        instruction: Instruction | None = None,
        memory: ConversationMemory | Iterable[ConversationMessage] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        **extra: Any,
    ) -> RealtimeConversationSessionScope:
        conversation: RealtimeConversation = ctx.state(cls)

        # prepare memory
        conversation_memory: ConversationMemory | None
        match memory:
            case None:
                conversation_memory = None

            case Memory() as memory:
                conversation_memory = memory

            case memory_messages:
                conversation_memory = cast(
                    ConversationMemory,
                    ConstantMemory(tuple(message.to_lmm_context() for message in memory_messages)),
                )

        return await conversation.preparing(
            instruction=Instruction.of(instruction),
            memory=conversation_memory,
            toolbox=Toolbox.of(tools),
            **extra,
        )

    preparing: RealtimeConversationPreparing = realtime_conversation_preparing
