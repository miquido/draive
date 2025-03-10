from collections.abc import AsyncIterable
from types import TracebackType
from typing import Protocol, Self, overload, runtime_checkable

from haiway import AsyncQueue, State, ctx
from haiway.context.state import StateContext

from draive.commons import Meta
from draive.multimodal import MultimodalContent
from draive.multimodal.content import Multimodal
from draive.parameters import DataModel

__all__ = [
    "Processing",
    "ProcessingContext",
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
    ) -> None: ...


class ProcessingContext:
    def __init__(
        self,
        context: StateContext,
        /,
        *,
        queue: AsyncQueue[ProcessingEvent],
    ) -> None:
        self._context: StateContext = context
        self._queue: AsyncQueue[ProcessingEvent] = queue

    def __enter__(self) -> AsyncIterable[ProcessingEvent]:
        self._context.__enter__()
        return self._queue

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._queue.finish()
        self._context.__exit__(
            exc_type,
            exc_val,
            exc_tb,
        )


class Processing(State):
    @classmethod
    def context(cls) -> ProcessingContext:
        queue = AsyncQueue[ProcessingEvent]()

        async def report_event(event: ProcessingEvent) -> None:
            queue.enqueue(event)

        return ProcessingContext(
            ctx.updated(Processing(event_reporting=report_event)),
            queue=queue,
        )

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
