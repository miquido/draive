from asyncio import Lock
from types import TracebackType
from typing import Any, Self

from draive.parameters import ParametrizedData, State
from draive.scope import ctx
from draive.types import MultimodalContent

__all__ = [
    "AgentState",
    "AgentScratchpad",
]


class AgentScratchpad(State):
    @classmethod
    def current(cls) -> Self:
        return ctx.state(cls)

    @classmethod
    def prepare(
        cls,
        content: MultimodalContent | None,
    ) -> Self:
        match content:
            case None:
                return cls(content=MultimodalContent.of())

            case item:
                return cls(content=item)

    content: MultimodalContent = MultimodalContent.of()

    def extended(
        self,
        content: MultimodalContent | None,
    ) -> Self:
        match content:
            case None:
                return self
            case item:
                return self.__class__(content=MultimodalContent.of(*self.content, item))


class AgentState[State: ParametrizedData]:
    def __init__(
        self,
        initial: State,
    ) -> None:
        self._lock: Lock = Lock()
        self._current: State = initial
        self._mutable_proxy: MutableAgentState[State] | None = None

    @property
    async def current(self) -> State:
        async with self._lock:
            return self._current

    @property
    def scratchpad(self) -> MultimodalContent:
        return AgentScratchpad.current().content

    async def __aenter__(self) -> "MutableAgentState[State]":
        await self._lock.__aenter__()
        assert self._mutable_proxy is None  # nosec: B101
        self._mutable_proxy = MutableAgentState(source=self)
        return self._mutable_proxy

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._mutable_proxy is not None  # nosec: B101
        self._mutable_proxy = None
        await self._lock.__aexit__(
            exc_type,
            exc_val,
            exc_tb,
        )


class MutableAgentState[State: ParametrizedData]:
    def __init__(
        self,
        source: AgentState[State],
    ) -> None:
        self._source: AgentState[State] = source

    # TODO: find a way to generate signature Based on ParametrizedData
    def update(
        self,
        **kwargs: Any,
    ) -> State:
        self._source._current = self._source._current.updated(**kwargs)  # pyright: ignore[reportPrivateUsage]
        return self._source._current  # pyright: ignore[reportPrivateUsage]
