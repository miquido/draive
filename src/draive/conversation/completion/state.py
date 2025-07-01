from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, final, overload

from haiway import State, as_tuple, ctx

from draive.conversation.completion.default import conversation_completion
from draive.conversation.completion.types import ConversationCompleting
from draive.conversation.types import (
    ConversationMemory,
    ConversationMessage,
    ConversationStreamElement,
)
from draive.instructions import Instruction
from draive.multimodal import Multimodal
from draive.tools import Tool, Toolbox
from draive.utils import Memory

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
        memory: ConversationMemory | Iterable[ConversationMessage] | None = None,
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
        memory: ConversationMemory | Iterable[ConversationMessage] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[ConversationStreamElement]: ...

    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: ConversationMessage | Multimodal,  # noqa: A002
        memory: ConversationMemory | Iterable[ConversationMessage] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[ConversationStreamElement] | ConversationMessage:
        conversation_message: ConversationMessage
        match input:
            case ConversationMessage():
                conversation_message = input

            case multimodal:
                conversation_message = ConversationMessage.user(multimodal)

        # prepare memory
        conversation_memory: ConversationMemory
        match memory:
            case None:
                conversation_memory = ConversationMemory.constant(())

            case Memory():
                conversation_memory = memory

            case messages:
                conversation_memory = ConversationMemory.constant(as_tuple(messages))

        if stream:
            return await ctx.state(cls).completing(
                instruction=Instruction.of(instruction),
                input=conversation_message,
                memory=conversation_memory,
                toolbox=Toolbox.of(tools),
                stream=True,
            )

        else:
            return await ctx.state(cls).completing(
                instruction=Instruction.of(instruction),
                input=conversation_message,
                memory=conversation_memory,
                toolbox=Toolbox.of(tools),
            )

    completing: ConversationCompleting = conversation_completion
