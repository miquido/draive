from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal, Self
from uuid import uuid4

from draive.parameters import DataModel, Field
from draive.types import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    MultimodalContent,
    MultimodalContentConvertible,
    ToolCallStatus,
)

__all__ = [
    "ConversationMessage",
    "ConversationMessageChunk",
    "ConversationResponseStream",
]


class ConversationMessage(DataModel):
    @classmethod
    def user(
        cls,
        content: MultimodalContent | MultimodalContentConvertible,
        identifier: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
    ) -> Self:
        return cls(
            identifier=identifier or uuid4().hex,
            role="user",
            author=author,
            created=created,
            content=MultimodalContent.of(content),
        )

    @classmethod
    def model(
        cls,
        content: MultimodalContent | MultimodalContentConvertible,
        identifier: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
    ) -> Self:
        return cls(
            identifier=identifier or uuid4().hex,
            role="model",
            author=author,
            created=created,
            content=MultimodalContent.of(content),
        )

    identifier: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "model"]
    author: str | None = None
    created: datetime | None = None
    content: MultimodalContent

    def as_lmm_context_element(self) -> LMMContextElement:
        match self.role:
            case "user":
                return LMMInput.of(self.content)

            case "model":
                return LMMCompletion.of(self.content)

    def __bool__(self) -> bool:
        return bool(self.content)


class ConversationMessageChunk(DataModel):
    identifier: str
    content: MultimodalContent

    def __bool__(self) -> bool:
        return bool(self.content)


ConversationResponseStream = AsyncIterator[ConversationMessageChunk | ToolCallStatus]
