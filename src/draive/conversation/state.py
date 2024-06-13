from collections.abc import Sequence

from draive.conversation.completion import ConversationCompletion
from draive.conversation.lmm import lmm_conversation_completion
from draive.conversation.model import ConversationMessage
from draive.parameters import State
from draive.types import Memory

__all__: list[str] = [
    "Conversation",
]


class Conversation(State):
    completion: ConversationCompletion = lmm_conversation_completion
    memory: Memory[Sequence[ConversationMessage], ConversationMessage] | None = None
