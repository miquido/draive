from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, overload

from draive.conversation.model import ConversationMessage, ConversationResponseStream
from draive.conversation.state import Conversation
from draive.helpers import ConstantMemory
from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.scope import ctx
from draive.types import Memory, MultimodalContent, MultimodalContentConvertible

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: Literal[True],
) -> ConversationResponseStream: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: Literal[False] = False,
) -> ConversationMessage: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: bool,
) -> ConversationResponseStream | ConversationMessage: ...


async def conversation_completion(
    *,
    instruction: Instruction | str,
    input: ConversationMessage | MultimodalContent | MultimodalContentConvertible,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    stream: bool = False,
) -> ConversationResponseStream | ConversationMessage:
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
    conversation_memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    match memory or conversation.memory:
        case None:
            conversation_memory = ConstantMemory([])

        case Memory() as memory:
            conversation_memory = memory

        case [*memory_messages]:
            conversation_memory = ConstantMemory(memory_messages)

    # prepare tools
    toolbox: Toolbox
    match tools:
        case None:
            toolbox = Toolbox()

        case Toolbox() as tools:
            toolbox = tools

        case [*tools]:
            toolbox = Toolbox(*tools)

    # request completion
    return await conversation.completion(
        instruction=instruction,
        message=message,
        memory=conversation_memory,
        toolbox=toolbox,
        stream=stream,
    )
