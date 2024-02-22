from draive.openai import openai_conversation_completion
from draive.scope import ScopeState
from draive.types import ConversationCompletion, ConversationMessage, Memory, Toolset

__all__: list[str] = [
    "Conversation",
]


class Conversation(ScopeState):
    completion: ConversationCompletion = openai_conversation_completion
    memory: Memory[ConversationMessage] | None = None
    toolset: Toolset | None = None
