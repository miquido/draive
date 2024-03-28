from typing import Literal

from draive.tools import ToolCallUpdate
from draive.types import ImageContent, Model, MultimodalContent

__all__ = [
    "LMMCompletionContent",
    "LMMCompletionMessage",
    "LMMCompletionStreamingUpdate",
]

LMMCompletionContent = MultimodalContent


class LMMCompletionMessage(Model):
    role: Literal["system", "assistant", "user"]
    content: LMMCompletionContent

    @property
    def has_media(self) -> bool:
        if isinstance(self.content, str):
            return False
        elif isinstance(self.content, ImageContent):
            return True
        elif isinstance(self.content, Model):
            return False
        else:
            return any(not isinstance(element, str) for element in self.content)

    @property
    def content_string(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, ImageContent):
            return ""
        elif isinstance(self.content, Model):
            return str(self.content)
        else:
            parts: list[str] = []
            for element in self.content:
                if isinstance(element, str):
                    parts.append(element)
                elif isinstance(self.content, Model):
                    parts.append(str(element))
                else:
                    continue  # skip media
            return "\n".join(parts)


LMMCompletionStreamingUpdate = LMMCompletionMessage | ToolCallUpdate
