from draive.conversation.state import Conversation
from draive.scope import ctx
from draive.types import ConversationMessage, Memory, StringConvertible, Toolset

__all__ = [
    "conversation_completion",
]


async def conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    conversation: Conversation = ctx.state(Conversation)
    return await conversation.completion(
        instruction=instruction,
        input=input,
        memory=memory or conversation.memory,
        toolset=toolset or conversation.toolset,
    )
