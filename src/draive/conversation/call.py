from typing import Literal, overload

from draive.conversation.state import Conversation
from draive.scope import ctx
from draive.types import (
    ConversationMessage,
    ConversationResponseStream,
    ConversationStreamingPart,
    Memory,
    StreamingProgressUpdate,
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
    stream: StreamingProgressUpdate[ConversationStreamingPart],
) -> ConversationMessage:
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
    stream: StreamingProgressUpdate[ConversationStreamingPart] | bool = False,
) -> ConversationResponseStream | ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)

    match stream:
        case False:
            return await conversation.completion(
                instruction=instruction,
                input=input,
                memory=memory or conversation.memory,
                toolset=toolset or conversation.toolset,
            )
        case True:
            return await conversation.completion(
                instruction=instruction,
                input=input,
                memory=memory or conversation.memory,
                toolset=toolset or conversation.toolset,
                stream=True,
            )
        case progress:
            return await conversation.completion(
                instruction=instruction,
                input=input,
                memory=memory or conversation.memory,
                toolset=toolset or conversation.toolset,
                stream=progress,
            )
