from typing import Protocol, overload, runtime_checkable

from haiway import State, ctx

from draive.commons import Meta
from draive.multimodal import MultimodalContent
from draive.multimodal.content import Multimodal
from draive.parameters import DataModel

__all__ = [
    "Processing",
    "ProcessingEvent",
    "ProcessingEventReporting",
    "ProcessingStateReading",
    "ProcessingStateWriting",
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


@runtime_checkable
class ProcessingStateReading(Protocol):
    async def __call__[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
    ) -> StateType | None: ...


@runtime_checkable
class ProcessingStateWriting(Protocol):
    async def __call__(
        self,
        value: DataModel | State,
        /,
    ) -> None: ...


async def _log_event_reporting(
    event: ProcessingEvent,
    /,
) -> None:
    ctx.log_info(f"ProcessingEvent:\n{event}")


async def _context_state_reading[StateType: DataModel | State](
    state: type[StateType],
    /,
    default: StateType | None = None,
) -> StateType | None:
    if issubclass(state, State):
        return ctx.state(state, default=default)

    else:
        return None


async def _ignored_state_writing(
    state: DataModel | State,
    /,
) -> None:
    ctx.log_warning(f"Ignoring processing state writing of {type(state)}")


class Processing(State):
    @overload
    @classmethod
    async def read[StateType: DataModel | State](
        cls,
        state: type[StateType],
        /,
    ) -> StateType | None: ...

    @overload
    @classmethod
    async def read[StateType: DataModel | State](
        cls,
        state: type[StateType],
        /,
        *,
        default: StateType,
    ) -> StateType: ...

    @classmethod
    async def read[StateType: DataModel | State](
        cls,
        state: type[StateType],
        /,
        *,
        default: StateType | None = None,
    ) -> StateType | None:
        current: StateType | None = await ctx.state(cls).state_reading(state)
        if current is None:
            return default

        else:
            return current

    @classmethod
    async def write(
        cls,
        state: DataModel | State,
        /,
    ) -> None:
        await ctx.state(cls).state_writing(state)

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

    event_reporting: ProcessingEventReporting = _log_event_reporting
    state_reading: ProcessingStateReading = _context_state_reading
    state_writing: ProcessingStateWriting = _ignored_state_writing
