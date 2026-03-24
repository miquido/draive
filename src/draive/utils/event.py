from datetime import UTC, datetime
from typing import Self

from haiway import Default, Meta, MetaValues, State

from draive.multimodal import Multimodal, MultimodalContent

__all__ = ("ProcessingEvent",)


class ProcessingEvent(State, serializable=True):
    """Structured event emitted while running a process."""

    @classmethod
    def of(
        cls,
        event: str,
        /,
        content: Multimodal = MultimodalContent.empty,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an event bound to the current processing context.

        Parameters
        ----------
        event : str
            Event name describing the emitted state transition or progress update.
        content : Multimodal, default=MultimodalContent.empty
            Event payload converted to multimodal content.
        meta : Meta | MetaValues | None, default=None
            Additional metadata merged with the context.

        Returns
        -------
        Self
            Event instance to be streamed during processing.
        """
        return cls(
            event=event,
            created=datetime.now(),
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    event: str
    created: datetime = Default(default_factory=lambda: datetime.now(UTC))
    content: MultimodalContent
    meta: Meta = Meta.empty
