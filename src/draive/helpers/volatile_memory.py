from collections.abc import Iterable
from typing import final

from draive.types import Memory

__all__ = [
    "ConstantMemory",
    "VolatileMemory",
    "VolatileAccumulativeMemory",
]


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


@final
class VolatileAccumulativeMemory[Item](Memory[list[Item], Item]):
    def __init__(
        self,
        items: Iterable[Item] | None = None,
        /,
        limit: int | None = None,
    ) -> None:
        self._items: list[Item] = list(items) if items else []
        self._limit: int = limit or 0

    async def recall(self) -> list[Item]:
        return self._items

    async def remember(
        self,
        *items: Item,
    ) -> None:
        if not items:
            return  # nothing to do

        self._items.extend(items)

        if self._limit > 0:
            self._items = self._items[-self._limit :]
