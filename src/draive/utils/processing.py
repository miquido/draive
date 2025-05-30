from collections.abc import Iterable
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable
from uuid import UUID, uuid4

from haiway import Default, MissingState, State, ctx

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.multimodal import MultimodalContent
from draive.multimodal.content import Multimodal
from draive.parameters import DataModel

__all__ = (
    "Processing",
    "ProcessingEvent",
    "ProcessingState",
)


class ProcessingEvent(DataModel):
    @classmethod
    def of(
        cls,
        name: str,
        /,
        *,
        identifier: UUID | None = None,
        content: Multimodal | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            name=name,
            content=MultimodalContent.of(content) if content is not None else None,
            meta=Meta.of(meta),
        )

    identifier: UUID = Default(factory=uuid4)
    name: str
    content: MultimodalContent | None = None
    meta: Meta = META_EMPTY


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
        required: Literal[True],
    ) -> StateType: ...

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
        required: bool = False,
    ) -> StateType | None:
        current: StateType | None = await ctx.state(cls).state_reading(state)
        if current is not None:
            return current

        elif default is not None:
            return default

        elif required:
            raise MissingState(f"{state.__qualname__} is not available within current processing")

        else:
            return None

    @classmethod
    async def write(
        cls,
        state: DataModel | State,
        /,
    ) -> None:
        await ctx.state(cls).state_writing(state)

    @overload
    @classmethod
    async def event(
        cls,
        event: ProcessingEvent,
        /,
    ) -> None: ...

    @overload
    @classmethod
    async def event(
        cls,
        /,
        *,
        identifier: UUID | None = None,
        name: str | None,
        content: Multimodal | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> None: ...

    @classmethod
    async def event(
        cls,
        event: ProcessingEvent | None = None,
        /,
        *,
        identifier: UUID | None = None,
        name: str | None = None,
        content: Multimodal | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        if event is not None:
            assert identifier is None and name is None  # nosec: B101
            return await ctx.state(cls).event_reporting(event)

        assert name is not None  # nosec: B101
        return await ctx.state(cls).event_reporting(
            ProcessingEvent.of(
                name,
                identifier=identifier,
                content=MultimodalContent.of(content) if content is not None else None,
                meta=Meta.of(meta),
            )
        )

    event_reporting: ProcessingEventReporting = _log_event_reporting
    state_reading: ProcessingStateReading = _context_state_reading
    state_writing: ProcessingStateWriting = _ignored_state_writing


class ProcessingState:
    def __init__(
        self,
        state: Iterable[DataModel | State] | None,
    ) -> None:
        self._state: dict[type[DataModel | State], Any]
        object.__setattr__(
            self,
            "_state",
            {type(element): element for element in state} if state else {},
        )

    async def read[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
        default: StateType | None = None,
    ) -> StateType | None:
        if state in self._state:
            return self._state[state]

        else:
            return default

    async def write(
        self,
        state: DataModel | State,
        /,
    ) -> None:
        self._state[type(state)] = state

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError("ProcessingState is frozen and can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError("ProcessingState is frozen and can't be modified")
