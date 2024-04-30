from typing import Literal

from draive.tools import ToolCallUpdate
from draive.types import (
    Model,
    MultimodalContent,
    has_media,
    multimodal_content_string,
)

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
        return has_media(self.content)

    @property
    def content_string(self) -> str:
        return multimodal_content_string(self.content)


LMMCompletionStreamingUpdate = LMMCompletionMessage | ToolCallUpdate
