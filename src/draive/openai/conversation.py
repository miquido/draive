from datetime import datetime

from draive.openai.chat import openai_chat_completion
from draive.types import ConversationMessage, Memory, StringConvertible, Toolset

__all__ = [
    "openai_conversation_completion",
]


async def openai_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    user_message: ConversationMessage
    if isinstance(input, ConversationMessage):
        user_message = input

    else:
        user_message = ConversationMessage(
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            author="user",
            content=input,
        )

    history: list[ConversationMessage]
    if memory:
        history = await memory.recall()
    else:
        history = []

    completion: str = await openai_chat_completion(
        instruction=instruction,
        input=user_message.content,
        history=history,
        toolset=toolset,
    )

    response: ConversationMessage = ConversationMessage(
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        author="assistant",
        content=completion,
    )

    if memory:
        await memory.remember(
            [
                *history,
                user_message,
                response,
            ],
        )
    return response
