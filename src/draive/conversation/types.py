from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable
from uuid import uuid4

from haiway import Default

from draive.commons import META_EMPTY, Meta
from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMStreamChunk,
)
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)
from draive.parameters import DataModel
from draive.tools import Toolbox
from draive.utils import Memory, ProcessingEvent

__all__ = (
    "ConversationCompleting",
    "ConversationElement",
    "ConversationMemory",
    "ConversationMessage",
)


class ConversationMessage(DataModel):
    @classmethod
    def user(
        cls,
        content: Multimodal,
        identifier: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            identifier=identifier or uuid4().hex,
            role="user",
            author=author,
            created=created,
            content=MultimodalContent.of(content),
            meta=meta if meta is not None else META_EMPTY,
        )

    @classmethod
    def model(
        cls,
        content: Multimodal,
        identifier: str | None = None,
        author: str | None = None,
        created: datetime | None = None,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            identifier=identifier or uuid4().hex,
            role="model",
            author=author,
            created=created,
            content=MultimodalContent.of(content),
            meta=meta if meta is not None else META_EMPTY,
        )

    identifier: str = Default(factory=lambda: uuid4().hex)
    role: Literal["user", "model"]
    author: str | None = None
    created: datetime | None = None
    content: MultimodalContent
    meta: Meta = Default(META_EMPTY)

    def as_lmm_context_element(self) -> LMMContextElement:
        match self.role:
            case "user":
                return LMMInput.of(self.content)

            case "model":
                return LMMCompletion.of(self.content)

    def __bool__(self) -> bool:
        return bool(self.content)


ConversationElement = ConversationMessage  # TODO: allow events/tool statuses
ConversationMemory = Memory[Sequence[ConversationElement], ConversationElement]


@runtime_checkable
class ConversationCompleting(Protocol):
    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: ConversationMessage | Multimodal,
        memory: ConversationMemory,
        toolbox: Toolbox,
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ConversationMessage: ...

    @overload
    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: ConversationMessage | Multimodal,
        memory: ConversationMemory,
        toolbox: Toolbox,
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk | ProcessingEvent]: ...

    async def __call__(
        self,
        *,
        instruction: Instruction | None,
        input: ConversationMessage | Multimodal,  # noqa: A002
        memory: ConversationMemory,
        toolbox: Toolbox,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk | ProcessingEvent] | ConversationMessage: ...
