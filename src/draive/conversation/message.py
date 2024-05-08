from datetime import datetime

from draive.lmm import LMMMessage, LMMStreamingUpdate

__all__ = [
    "ConversationMessage",
    "ConversationStreamingUpdate",
]


class ConversationMessage(LMMMessage):
    author: str | None = None
    created: datetime | None = None


ConversationStreamingUpdate = LMMStreamingUpdate
