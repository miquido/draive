from collections.abc import Awaitable, Callable, Iterable
from typing import Generic, Protocol, TypeVar, final, runtime_checkable

__all__ = [
    "Memory",
    "ReadOnlyMemory",
]

_MemoryElement = TypeVar(
    "_MemoryElement",
    bound=object,
)


@runtime_checkable
class Memory(Protocol, Generic[_MemoryElement]):
    async def recall(self) -> list[_MemoryElement]:
        ...

    async def remember(
        self,
        elements: Iterable[_MemoryElement],
        /,
    ) -> None:
        ...


@final
class ReadOnlyMemory(Generic[_MemoryElement]):
    def __init__(
        self,
        elements: (Callable[[], Awaitable[list[_MemoryElement]]] | Iterable[_MemoryElement]),
    ) -> None:
        self._elements: Callable[[], Awaitable[list[_MemoryElement]]]
        if callable(elements):
            self._elements = elements
        else:
            messages_list: list[_MemoryElement] = list(elements)

            async def constant() -> list[_MemoryElement]:
                return messages_list

            self._elements = constant

    async def recall(self) -> list[_MemoryElement]:
        return await self._elements()

    async def remember(
        self,
        *elements: _MemoryElement,
    ) -> None:
        pass  # ignore
