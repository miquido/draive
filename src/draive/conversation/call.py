from typing import Literal, overload

from draive.conversation.completion import ConversationCompletionStream
from draive.conversation.message import (
    ConversationMessage,
    ConversationMessageContent,
    ConversationStreamingUpdate,
)
from draive.conversation.state import Conversation
from draive.scope import ctx
from draive.tools import Toolbox
from draive.types import Memory, UpdateSend

__all__ = [
    "conversation_completion",
]


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
    stream: Literal[True],
) -> ConversationCompletionStream:
    ...


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
    stream: UpdateSend[ConversationStreamingUpdate],
) -> ConversationMessage:
    ...


@overload
async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
) -> ConversationMessage:
    ...


async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | ConversationMessageContent,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    tools: Toolbox | None = None,
    stream: UpdateSend[ConversationStreamingUpdate] | bool = False,
) -> ConversationCompletionStream | ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)

    match stream:
        case False:
            return await conversation.completion(
                instruction=instruction,
                input=input,
                memory=memory or conversation.memory,
                tools=tools or conversation.tools,
            )
        case True:
            return await conversation.completion(
                instruction=instruction,
                input=input,
                memory=memory or conversation.memory,
                tools=tools or conversation.tools,
                stream=True,
            )
        case progress:
            return await conversation.completion(
                instruction=instruction,
                input=input,
                memory=memory or conversation.memory,
                tools=tools or conversation.tools,
                stream=progress,
            )
