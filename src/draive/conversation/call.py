from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal, overload

from haiway import ctx

from draive.conversation.state import Conversation
from draive.conversation.types import ConversationElement, ConversationMemory, ConversationMessage
from draive.instructions import Instruction
from draive.lmm import LMMStreamChunk
from draive.multimodal import Multimodal
from draive.prompts import Prompt
from draive.tools import AnyTool, Toolbox
from draive.utils import Memory, ProcessingEvent

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Prompt | Multimodal,
    memory: ConversationMemory | Iterable[ConversationElement] | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent]: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Prompt | Multimodal,
    memory: ConversationMemory | Iterable[ConversationElement] | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Prompt | Multimodal,  # noqa: A002
    memory: ConversationMemory | Iterable[ConversationElement] | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
    stream: bool = False,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk | ProcessingEvent] | ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)

    # prepare memory
    conversation_memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    match memory or conversation.memory:
        case None:
            conversation_memory = Memory[
                Sequence[ConversationMessage],
                ConversationMessage,
            ].constant(recalled=())

        case Memory() as memory:
            conversation_memory = memory

        case memory_messages:
            conversation_memory = Memory[
                Sequence[ConversationMessage],
                ConversationMessage,
            ].constant(recalled=tuple(memory_messages))

    # request completion
    return await conversation.completion(
        instruction=instruction,
        input=input,
        memory=conversation_memory,
        toolbox=Toolbox.of(tools),
        stream=stream,
    )
