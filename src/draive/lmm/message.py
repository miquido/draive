from typing import Literal

from draive.tools import ToolCallUpdate
from draive.types import (
    Model,
    MultimodalContent,
    has_media,
    multimodal_content_string,
)

__all__ = [
    "LMMMessage",
    "LMMStreamingUpdate",
]


class LMMMessage(Model):
    role: Literal["system", "assistant", "user"]
    content: MultimodalContent

    @property
    def has_media(self) -> bool:
        return has_media(self.content)

    @property
    def content_string(self) -> str:
        return multimodal_content_string(self.content)


LMMStreamingUpdate = LMMMessage | ToolCallUpdate
