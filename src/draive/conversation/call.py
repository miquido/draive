from typing import Literal, overload

from draive.conversation.state import Conversation
from draive.scope import ctx
from draive.types import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationResponseStream,
    ConversationStreamingUpdate,
    Memory,
    ProgressUpdate,
    Toolset,
)

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate],
) -> ConversationMessage:
    ...


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    ...


async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: ProgressUpdate[ConversationStreamingUpdate] | bool = False,
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
