from draive.lmm import LMMCompletionContent, LMMCompletionMessage, LMMCompletionStreamingUpdate

__all__ = [
    "ConversationMessage",
    "ConversationMessageContent",
    "ConversationStreamingUpdate",
]


class ConversationMessage(LMMCompletionMessage):
    author: str | None = None
    timestamp: str | None = None


ConversationMessageContent = LMMCompletionContent
ConversationStreamingUpdate = LMMCompletionStreamingUpdate
