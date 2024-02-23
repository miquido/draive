from typing import Literal, overload

from draive.conversation.state import Conversation
from draive.scope import ctx
from draive.types import (
    ConversationMessage,
    ConversationResponseStream,
    Memory,
    StringConvertible,
    Toolset,
)

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    ...


async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: bool = False,
) -> ConversationResponseStream | ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)

    if stream:  # pyright does not get the type right here, have to use literals
        return await conversation.completion(
            instruction=instruction,
            input=input,
            memory=memory or conversation.memory,
            toolset=toolset or conversation.toolset,
            stream=True,
        )

    else:
        return await conversation.completion(
            instruction=instruction,
            input=input,
            memory=memory or conversation.memory,
            toolset=toolset or conversation.toolset,
        )
