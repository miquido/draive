from draive.types.state import State
from draive.types.string import StringConvertible

__all__ = [
    "ConversationMessage",
]


class ConversationMessage(State):
    author: str
    content: StringConvertible
    timestamp: str | None = None

    @property
    def content_str(self) -> str:
        return str(self.content)
