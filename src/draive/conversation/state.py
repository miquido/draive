from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal, final, overload

from haiway import State, ctx

from draive.conversation.default import conversation_completion
from draive.conversation.types import (
    ConversationCompleting,
    ConversationElement,
    ConversationMemory,
    ConversationMessage,
)
from draive.helpers import ConstantMemory
from draive.instructions import Instruction
from draive.lmm import LMMStreamChunk
from draive.multimodal import Multimodal
from draive.tools import Tool, Toolbox
from draive.utils import Memory, ProcessingEvent

__all__ = ("Conversation",)


@final
class Conversation(State):
    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: ConversationMessage | Multimodal,
        memory: ConversationMemory | Iterable[ConversationElement] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: ConversationMessage | Multimodal,
        memory: ConversationMemory | Iterable[ConversationElement] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk | ProcessingEvent]: ...

    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: ConversationMessage | Multimodal,  # noqa: A002
        memory: ConversationMemory | Iterable[ConversationElement] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk | ProcessingEvent] | ConversationMessage:
        conversation: Conversation = ctx.state(cls)

        # prepare memory
        conversation_memory: Memory[Sequence[ConversationMessage], ConversationMessage]
        match memory or conversation.memory:
            case None:
                conversation_memory = ConstantMemory(recalled=())

            case Memory() as memory:
                conversation_memory = memory

            case memory_messages:
                conversation_memory = ConstantMemory(recalled=tuple(memory_messages))

        if stream:
            return await conversation.completing(
                instruction=Instruction.of(instruction),
                input=input,
                memory=conversation_memory,
                toolbox=Toolbox.of(tools),
                stream=True,
            )

        else:
            return await conversation.completing(
                instruction=Instruction.of(instruction),
                input=input,
                memory=conversation_memory,
                toolbox=Toolbox.of(tools),
            )

    completing: ConversationCompleting = conversation_completion
    memory: Memory[Sequence[ConversationMessage], ConversationMessage] | None = None
