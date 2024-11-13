from collections.abc import AsyncIterator, Iterable, Mapping
from datetime import datetime
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable
from uuid import uuid4

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMStreamChunk,
    Toolbox,
)
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)
from draive.parameters import DataModel, Field
from draive.utils import Memory

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
        meta: Mapping[str, str | float | int | bool | None] | None = None,
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
        meta: Mapping[str, str | float | int | bool | None] | None = None,
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
    meta: Mapping[str, str | float | int | bool | None] | None = None

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
    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        message: ConversationMessage,
        memory: Memory[Iterable[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk]: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        message: ConversationMessage,
        memory: Memory[Iterable[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        message: ConversationMessage,
        memory: Memory[Iterable[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: bool,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk] | ConversationMessage: ...

    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        message: ConversationMessage,
        memory: Memory[Iterable[ConversationMessage], ConversationMessage],
        toolbox: Toolbox,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk] | ConversationMessage: ...
