from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Final

from haiway import ctx

from draive.conversation.state import Conversation
from draive.conversation.types import ConversationMessage
from draive.helpers import ConstantMemory
from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.types import Memory, Multimodal, MultimodalContent

__all__ = [
    "conversation_completion",
]


async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Multimodal,  # noqa: A002
    memory: Memory[Iterable[ConversationMessage], ConversationMessage]
    | Iterable[ConversationMessage]
    | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
) -> ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)

    # prepare message
    message: ConversationMessage
    match input:
        case ConversationMessage() as conversation_message:
            if guardrails := conversation.guardrails:
                message = conversation_message.updated(
                    content=await guardrails(conversation_message.content),
                )

            else:
                message = conversation_message

        case content:
            if guardrails := conversation.guardrails:
                message = ConversationMessage(
                    role="user",
                    created=datetime.now(UTC),
                    content=await guardrails(MultimodalContent.of(content)),
                )

            else:
                message = ConversationMessage(
                    role="user",
                    created=datetime.now(UTC),
                    content=MultimodalContent.of(content),
                )

    # prepare memory
    conversation_memory: Memory[Iterable[ConversationMessage], ConversationMessage]
    match memory or conversation.memory:
        case None:
            conversation_memory = _EMPTY_MEMORY

        case Memory() as memory:
            conversation_memory = memory

        case memory_messages:
            conversation_memory = ConstantMemory(memory_messages)

    # request completion
    return await conversation.completion(
        instruction=instruction,
        message=message,
        memory=conversation_memory,
        toolbox=Toolbox.of(tools),
    )


_EMPTY_MEMORY: Final[ConstantMemory[Any, Any]] = ConstantMemory(tuple[Any]())
