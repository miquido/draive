from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Literal, Self
from uuid import UUID, uuid4

from haiway import Default

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.lmm import LMMCompletion, LMMContextElement, LMMInput
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel
from draive.utils import Memory

__all__ = (
    "ConversationElement",
    "ConversationMessage",
    "ConversationMessageChunk",
    "ConversationStreamElement",
)


class ConversationMessageChunk(DataModel):
    @classmethod
    def user(
        cls,
        content: Multimodal,
        *,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
        eod: bool = False,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            role="user",
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content),
            eod=eod,
            meta=Meta.of(meta),
        )

    @classmethod
    def model(
        cls,
        content: Multimodal,
        *,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
        eod: bool = False,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            role="model",
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content),
            eod=eod,
            meta=Meta.of(meta),
        )

    type: Literal["message_chunk"] = "message_chunk"
    identifier: UUID = Default(factory=uuid4)
    role: Literal["user", "model"]
    created: datetime = Default(factory=datetime.now)
    content: MultimodalContent
    eod: bool = False
    meta: Meta = META_EMPTY

    def __bool__(self) -> bool:
        return bool(self.content)


class ConversationMessage(DataModel):
    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[ConversationMessageChunk],
        /,
    ) -> Sequence[Self]:
        messages: list[Self] = []
        current: Self | None = None
        for chunk in chunks:
            if current is None:
                current = cls(
                    role=chunk.role,
                    created=chunk.created,
                    content=chunk.content,
                    meta=chunk.meta,
                )

            elif current.role == chunk.role:
                current = current.updated(
                    content=current.content.appending(chunk.content),
                    meta=current.meta.merged_with(chunk.meta),
                )

            else:
                messages.append(current)
                current = cls(
                    role=chunk.role,
                    created=chunk.created,
                    content=chunk.content,
                    meta=chunk.meta,
                )

            if chunk.eod:
                messages.append(current)
                current = None

        if current is not None:
            messages.append(current)

        return messages

    @classmethod
    def user(
        cls,
        content: Multimodal,
        *,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            role="user",
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    @classmethod
    def model(
        cls,
        content: Multimodal,
        *,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            role="model",
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    type: Literal["message"] = "message"
    identifier: UUID = Default(factory=uuid4)
    role: Literal["user", "model"]
    created: datetime = Default(factory=datetime.now)
    content: MultimodalContent
    meta: Meta = META_EMPTY

    def as_lmm_context_element(self) -> LMMContextElement:
        match self.role:
            case "user":
                return LMMInput.of(self.content)

            case "model":
                return LMMCompletion.of(self.content)

    def __bool__(self) -> bool:
        return bool(self.content)


class ConversationEvent(DataModel):
    @classmethod
    def of(
        cls,
        name: str,
        *,
        content: Multimodal | None = None,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            name=name,
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content)
            if content is not None
            else MultimodalContent.empty,
            meta=Meta.of(meta),
        )

    type: Literal["event"] = "event"
    identifier: UUID = Default(factory=uuid4)
    name: str
    created: datetime = Default(factory=datetime.now)
    content: MultimodalContent = MultimodalContent.empty
    meta: Meta = META_EMPTY


ConversationElement = ConversationMessage | ConversationEvent
ConversationMemory = Memory[Sequence[ConversationElement], ConversationElement]
ConversationStreamElement = ConversationMessageChunk | ConversationEvent
RealtimeConversationMemory = Memory[Sequence[ConversationElement], ConversationStreamElement]
