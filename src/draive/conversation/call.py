from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Final, Literal, overload

from haiway import ctx

from draive.conversation.state import Conversation
from draive.conversation.types import ConversationMessage
from draive.instructions import Instruction
from draive.lmm import AnyTool, LMMStreamChunk, Toolbox
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)
from draive.utils import Memory

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Multimodal,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
    stream: Literal[True],
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk]: ...


@overload
async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Multimodal,
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
    stream: Literal[False] = False,
    **extra: Any,
) -> ConversationMessage: ...


async def conversation_completion(
    *,
    instruction: Instruction | str | None = None,
    input: ConversationMessage | Multimodal,  # noqa: A002
    memory: Memory[Sequence[ConversationMessage], ConversationMessage]
    | Sequence[ConversationMessage]
    | None = None,
    tools: Toolbox | Iterable[AnyTool] | None = None,
    stream: bool = False,
    **extra: Any,
) -> AsyncIterator[LMMStreamChunk] | ConversationMessage:
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
            conversation_memory = _EMPTY_MEMORY

        case Memory() as memory:
            conversation_memory = memory

        case memory_messages:
            conversation_memory = Memory[
                Sequence[ConversationMessage],
                ConversationMessage,
            ].constant(recalled=memory_messages)

    # request completion
    return await conversation.completion(
        instruction=instruction,
        message=message,
        memory=conversation_memory,
        toolbox=Toolbox.out_of(tools),
        stream=stream,
    )


_EMPTY_MEMORY: Final[
    Memory[
        Sequence[ConversationMessage],
        ConversationMessage,
    ]
] = Memory[
    Sequence[ConversationMessage],
    ConversationMessage,
].constant(recalled=())
