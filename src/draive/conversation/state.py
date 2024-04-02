from draive.conversation.completion import ConversationCompletion
from draive.conversation.lmm import lmm_conversation_completion
from draive.conversation.message import (
    ConversationMessage,
)
from draive.tools import Toolbox
from draive.types import Memory, State

__all__: list[str] = [
    "Conversation",
]


class Conversation(State):
    completion: ConversationCompletion = lmm_conversation_completion
    memory: Memory[ConversationMessage] | None = None
    tools: Toolbox | None = None
