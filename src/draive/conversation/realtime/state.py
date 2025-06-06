from collections.abc import Iterable
from typing import Any, final

from haiway import State, ctx

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.realtime.types import (
    RealtimeConversationPreparing,
    RealtimeConversationSessionScope,
)
from draive.conversation.types import ConversationMessage
from draive.instructions import Instruction
from draive.lmm import LMMMemory
from draive.tools import Tool, Toolbox

__all__ = ("RealtimeConversation",)


@final
class RealtimeConversation(State):
    @classmethod
    async def prepare(
        cls,
        *,
        instruction: Instruction | None = None,
        memory: LMMMemory | Iterable[ConversationMessage] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        **extra: Any,
    ) -> RealtimeConversationSessionScope:
        conversation: RealtimeConversation = ctx.state(cls)

        # prepare memory
        conversation_memory: LMMMemory | None
        match memory:
            case None:
                conversation_memory = None

            case LMMMemory() as memory:
                conversation_memory = memory

            case memory_messages:
                conversation_memory = LMMMemory.constant(
                    tuple(message.to_lmm_context() for message in memory_messages)
                )

        return await conversation.preparing(
            instruction=Instruction.of(instruction),
            memory=conversation_memory,
            toolbox=Toolbox.of(tools),
            **extra,
        )

    preparing: RealtimeConversationPreparing = realtime_conversation_preparing
