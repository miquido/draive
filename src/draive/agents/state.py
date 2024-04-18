from asyncio import Lock
from collections.abc import Callable
from typing import Any

from draive.parameters import ParametrizedData
from draive.types import MultimodalContent, MultimodalContentItem, merge_multimodal_content

__all__ = [
    "AgentState",
]


class AgentState[State: ParametrizedData]:
    def __init__(
        self,
        initial: State,
        scratchpad: MultimodalContent | None = None,
    ) -> None:
        self._lock: Lock = Lock()
        self._current: State = initial
        self._scratchpad: tuple[MultimodalContentItem, ...]
        match scratchpad:
            case None:
                self._scratchpad = ()
            case [*items]:
                self._scratchpad = tuple(items)
            case item:
                self._scratchpad = (item,)

    @property
    async def current(self) -> State:
        async with self._lock:
            return self._current

    @property
    async def scratchpad(self) -> MultimodalContent:
        async with self._lock:
            return self._scratchpad

    async def extend_scratchpad(
        self,
        content: MultimodalContent,
    ) -> MultimodalContent:
        async with self._lock:
            self._scratchpad = merge_multimodal_content(self._scratchpad, content)
            return self._scratchpad

    async def apply(
        self,
        patch: Callable[[State], State],
    ) -> State:
        async with self._lock:
            self._current = patch(self._current)
            return self._current

    # TODO: find a way to generate signature Based on ParametrizedData
    async def update(
        self,
        **kwargs: Any,
    ) -> State:
        async with self._lock:
            self._current = self._current.updated(**kwargs)
            return self._current
