from abc import ABC, abstractmethod
from typing import final

__all__ = [
    "Memory",
    "ConstantMemory",
    "VolatileMemory",
]


class Memory[Recalled, Remembered](ABC):
    @abstractmethod
    async def recall(self) -> Recalled: ...

    @abstractmethod
    async def remember(
        self,
        *items: Remembered,
    ) -> None: ...


@final
class ConstantMemory[Recalled, Remembered](Memory[Recalled, Remembered]):
    def __init__(
        self,
        item: Recalled,
        /,
    ) -> None:
        self._item: Recalled = item

    async def recall(self) -> Recalled:
        return self._item

    async def remember(
        self,
        *items: Remembered,
    ) -> None:
        pass  # ignore


@final
class VolatileMemory[Item](Memory[Item, Item]):
    def __init__(
        self,
        item: Item,
        /,
    ) -> None:
        self._item: Item = item

    async def recall(self) -> Item:
        return self._item

    async def remember(
        self,
        *items: Item,
    ) -> None:
        if items:
            self._item = items[-1]
