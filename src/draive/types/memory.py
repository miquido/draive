from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from typing import Generic, TypeVar, final

__all__ = [
    "Memory",
    "ReadOnlyMemory",
]

_MemoryElement = TypeVar(
    "_MemoryElement",
    bound=object,
)


class Memory(ABC, Generic[_MemoryElement]):
    @abstractmethod
    async def recall(self) -> list[_MemoryElement]:
        ...

    @abstractmethod
    async def remember(
        self,
        elements: Iterable[_MemoryElement],
        /,
    ) -> None:
        ...


@final
class ReadOnlyMemory(Memory[_MemoryElement]):
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
        elements: Iterable[_MemoryElement],
        /,
    ) -> None:
        pass  # ignore
