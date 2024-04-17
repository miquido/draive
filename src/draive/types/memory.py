from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from typing import final

__all__ = [
    "Memory",
    "ReadOnlyMemory",
]


class Memory[Element](ABC):
    @abstractmethod
    async def recall(self) -> list[Element]: ...

    @abstractmethod
    async def remember(
        self,
        elements: Iterable[Element],
        /,
    ) -> None: ...


@final
class ReadOnlyMemory[Element](Memory[Element]):
    def __init__(
        self,
        elements: (Callable[[], Awaitable[list[Element]]] | Iterable[Element]),
    ) -> None:
        self._elements: Callable[[], Awaitable[list[Element]]]
        if callable(elements):
            self._elements = elements
        else:
            messages_list: list[Element] = list(elements)

            async def constant() -> list[Element]:
                return messages_list

            self._elements = constant

    async def recall(self) -> list[Element]:
        return await self._elements()

    async def remember(
        self,
        elements: Iterable[Element],
        /,
    ) -> None:
        pass  # ignore
