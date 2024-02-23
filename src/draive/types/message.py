from typing import Literal

from draive.types.model import Model
from draive.types.string import StringConvertible

__all__ = [
    "ConversationMessage",
    "ConversationStreamingActionStatus",
    "ConversationStreamingAction",
    "ConversationStreamingPart",
]


class ConversationMessage(Model):
    author: str
    content: StringConvertible
    timestamp: str | None = None

    @property
    def content_str(self) -> str:
        return str(self.content)


class ConversationStreamingActionStatus(Model):
    status: Literal["INITIAL", "PROGRESS", "COMPLETED"]
    message: str | None = None


class ConversationStreamingAction(Model):
    id: str
    name: str
    status: ConversationStreamingActionStatus


class ConversationStreamingPart(Model):
    actions: list[ConversationStreamingAction] | None
    message: ConversationMessage | None
