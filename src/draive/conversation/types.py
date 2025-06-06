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
    "ConversationEvent",
    "ConversationMemory",
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
        message_identifier: UUID,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
        eod: bool = False,
    ) -> Self:
        return cls(
            message_identifier=message_identifier,
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
        message_identifier: UUID,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
        eod: bool = False,
    ) -> Self:
        return cls(
            message_identifier=message_identifier,
            identifier=identifier if identifier is not None else uuid4(),
            role="model",
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content),
            eod=eod,
            meta=Meta.of(meta),
        )

    type: Literal["message_chunk"] = "message_chunk"
    message_identifier: UUID
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
        for chunk in chunks:  # assuming correct order
            if current is None:
                current = cls(
                    identifier=chunk.message_identifier,
                    role=chunk.role,
                    created=chunk.created,
                    content=chunk.content,
                    meta=chunk.meta,
                )

            elif current.role == chunk.role:
                assert current.created < chunk.created  # nosec: B101
                assert current.identifier == chunk.message_identifier  # nosec: B101

                current = current.with_chunk(chunk, merge_meta=True)

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
    def from_lmm_context(
        cls,
        element: LMMInput | LMMCompletion,
        /,
    ) -> Self:
        match element:
            case LMMInput():
                return cls.user(
                    element.content,
                    identifier=element.meta.identifier,
                    created=element.meta.creation,
                    meta=element.meta.excluding("kind", "identifier", "creation"),
                )

            case LMMCompletion():
                return cls.model(
                    element.content,
                    identifier=element.meta.identifier,
                    created=element.meta.creation,
                    meta=element.meta.excluding("kind", "identifier", "creation"),
                )

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

    identifier: UUID = Default(factory=uuid4)
    role: Literal["user", "model"]
    created: datetime = Default(factory=datetime.now)
    content: MultimodalContent
    meta: Meta = META_EMPTY

    def to_lmm_context(
        self,
    ) -> LMMContextElement:
        match self.role:
            case "user":
                return LMMInput.of(
                    self.content,
                    meta=self.meta.updated(
                        # using predefined meta keys
                        kind="message",
                        identifier=self.identifier.hex,
                        creation=self.created.isoformat(),
                    ),
                )

            case "model":
                return LMMCompletion.of(
                    self.content,
                    meta=self.meta.updated(
                        # using predefined meta keys
                        kind="message",
                        identifier=self.identifier.hex,
                        creation=self.created.isoformat(),
                    ),
                )

    def __bool__(self) -> bool:
        return bool(self.content)

    def with_chunk(
        self,
        chunk: ConversationMessageChunk,
        /,
        merge_meta: bool = False,
    ) -> Self:
        assert self.identifier == chunk.message_identifier  # nosec: B101
        return self.__class__(
            identifier=self.identifier,
            role=self.role,
            created=self.created,
            content=self.content.appending(chunk.content),
            meta=self.meta.merged_with(chunk.meta) if merge_meta else self.meta,
        )


class ConversationEvent(DataModel):
    @classmethod
    def of(
        cls,
        category: str,
        *,
        content: Multimodal | None = None,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            category=category,
            created=created if created is not None else datetime.now(),
            content=MultimodalContent.of(content)
            if content is not None
            else MultimodalContent.empty,
            meta=Meta.of(meta),
        )

    identifier: UUID = Default(factory=uuid4)
    category: str
    created: datetime = Default(factory=datetime.now)
    content: MultimodalContent = MultimodalContent.empty
    meta: Meta = META_EMPTY


ConversationElement = ConversationMessage | ConversationEvent
ConversationStreamElement = ConversationMessageChunk | ConversationEvent

ConversationMemory = Memory[Sequence[ConversationMessage], ConversationMessage]
