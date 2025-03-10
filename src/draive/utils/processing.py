from typing import Protocol, Self, overload, runtime_checkable

from haiway import State, ctx

from draive.commons import Meta
from draive.multimodal import MultimodalContent
from draive.multimodal.content import Multimodal
from draive.parameters import DataModel

__all__ = [
    "Processing",
    "ProcessingEvent",
    "ProcessingEventReporting",
]


class ProcessingEvent(DataModel):
    identifier: str
    name: str
    content: MultimodalContent | None = None
    meta: Meta | None = None


@runtime_checkable
class ProcessingEventReporting(Protocol):
    async def __call__(
        self,
        event: ProcessingEvent,
        /,
    ) -> None: ...


class Processing(State):
    @classmethod
    def current(cls) -> Self:
        return ctx.state(cls)

    @overload
    @classmethod
    async def report_event(
        cls,
        event: ProcessingEvent,
        /,
    ) -> None: ...

    @overload
    @classmethod
    async def report_event(
        cls,
        /,
        *,
        identifier: str,
        name: str | None,
        content: Multimodal | None | None = None,
        meta: Meta | None = None,
    ) -> None: ...

    @classmethod
    async def report_event(
        cls,
        event: ProcessingEvent | None = None,
        /,
        *,
        identifier: str | None = None,
        name: str | None = None,
        content: Multimodal | None | None = None,
        meta: Meta | None = None,
    ) -> None:
        if event is not None:
            assert identifier is None and name is None  # nosec: B101
            return await ctx.state(cls).event_reporting(event)

        assert identifier is not None and name is not None  # nosec: B101
        return await ctx.state(cls).event_reporting(
            ProcessingEvent(
                identifier=identifier,
                name=name,
                content=MultimodalContent.of(content) if content is not None else None,
                meta=meta,
            )
        )

    event_reporting: ProcessingEventReporting
