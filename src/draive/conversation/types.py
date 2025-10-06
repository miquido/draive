from datetime import UTC, datetime
from typing import Literal, Self, final
from uuid import UUID, uuid4

from haiway import META_EMPTY, Meta, MetaValues

from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel

__all__ = (
    "ConversationEvent",
    "ConversationInputChunk",
    "ConversationMessage",
    "ConversationOutputChunk",
)


@final
class ConversationInputChunk(DataModel):
    """Streaming input chunk for a conversation session.

    Attributes
    ----------
    created : datetime
        Timestamp when the chunk was created (UTC).
    content : MultimodalContent
        Incremental multimodal content payload.
    eod : bool
        End-of-data marker for the current input stream.
    """

    @classmethod
    def of(
        cls,
        content: Multimodal,
        *,
        eod: bool = False,
    ) -> Self:
        """Create a conversation input chunk from content.

        Parameters
        ----------
        content : Multimodal
            One or more multimodal elements.
        eod : bool, optional
            Marks the end of the input stream.
        """
        return cls(
            created=datetime.now(UTC),
            content=MultimodalContent.of(content),
            eod=eod,
        )

    created: datetime
    content: MultimodalContent
    eod: bool

    def __bool__(self) -> bool:
        """Return ``True`` when any content is present."""
        return bool(self.content)


@final
class ConversationOutputChunk(DataModel):
    """Streaming output chunk produced during a conversation session.

    Attributes
    ----------
    created : datetime
        Timestamp when the chunk was created (UTC).
    content : MultimodalContent
        Incremental multimodal content payload.
    eod : bool
        End-of-data marker for the current output stream.
    """

    @classmethod
    def of(
        cls,
        content: Multimodal,
        *,
        eod: bool = False,
    ) -> Self:
        """Create a conversation output chunk from content."""
        return cls(
            created=datetime.now(UTC),
            content=MultimodalContent.of(content),
            eod=eod,
        )

    created: datetime
    content: MultimodalContent
    eod: bool

    def __bool__(self) -> bool:
        """Return ``True`` when any content is present."""
        return bool(self.content)


@final
class ConversationEvent(DataModel):
    """Streaming event produced during a conversation session.

    Attributes
    ----------
    created : datetime
        Timestamp when the event was created (UTC).
    category : str
        Name of the event type.
    content : MultimodalContent
        Optional, multimodal content payload.
    meta : Meta
        Additional metadata.
    """

    @classmethod
    def of(
        cls,
        category: str,
        *,
        content: Multimodal = MultimodalContent.empty,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a conversation event from content."""
        return cls(
            created=datetime.now(UTC),
            category=category,
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    created: datetime
    category: str
    content: MultimodalContent
    meta: Meta = META_EMPTY


@final
class ConversationMessage(DataModel):
    """Represents a single user or model message with multimodal content.

    Attributes
    ----------
    identifier : UUID
        Unique message identifier.
    role : {"user", "model"}
        Message author.
    created : datetime
        Creation timestamp (UTC).
    content : MultimodalContent
        Message content.
    meta : Meta
        Additional metadata.
    """

    @classmethod
    def user(
        cls,
        content: Multimodal,
        *,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a user-authored message."""
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            role="user",
            created=created if created is not None else datetime.now(UTC),
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
        """Create a model-authored message."""
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            role="model",
            created=created if created is not None else datetime.now(UTC),
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    identifier: UUID
    role: Literal["user", "model"]
    created: datetime
    content: MultimodalContent
    meta: Meta

    def __bool__(self) -> bool:
        """Return ``True`` when message has any content."""
        return bool(self.content)
