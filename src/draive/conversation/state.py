from collections.abc import Sequence

from haiway import State

from draive.conversation.default import default_conversation_completion
from draive.conversation.types import ConversationCompletion, ConversationMessage
from draive.safeguards import ContentGuardrails
from draive.utils import Memory

__all__: list[str] = [
    "Conversation",
]


class Conversation(State):
    completion: ConversationCompletion = default_conversation_completion
    memory: Memory[Sequence[ConversationMessage], ConversationMessage] | None = None
    guardrails: ContentGuardrails | None = None
