from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal, Protocol, Self, runtime_checkable
from uuid import uuid4

from draive.instructions import Instruction
from draive.lmm import Toolbox
from draive.parameters import DataModel, Field
from draive.types import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    Memory,
    Multimodal,
    MultimodalContent,
)

__all__ = [
    "ConversationCompletion",
    "ConversationMessage",
]


class ConversationMessage(DataModel):
    @classmethod
    def user(
        cls,
        content: Multimodal,
        identifier: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
        meta: dict[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            identifier=identifier or uuid4().hex,
            role="user",
            author=author,
            created=created,
            content=MultimodalContent.of(content),
            meta=meta,
        )

    @classmethod
    def model(
        cls,
        content: Multimodal,
        identifier: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
        meta: dict[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            identifier=identifier or uuid4().hex,
            role="model",
            author=author,
            created=created,
            content=MultimodalContent.of(content),
            meta=meta,
        )

    identifier: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "model"]
    author: str | None = None
    created: datetime | None = None
    content: MultimodalContent
    meta: dict[str, str | float | int | bool | None] | None = None

    def as_lmm_context_element(self) -> LMMContextElement:
        match self.role:
            case "user":
                return LMMInput.of(self.content)

            case "model":
                return LMMCompletion.of(self.content)

    def __bool__(self) -> bool:
        return bool(self.content)


@runtime_checkable
class ConversationCompletion(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        message: ConversationMessage,
        memory: Memory[Iterable[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        **extra: Any,
    ) -> ConversationMessage: ...
