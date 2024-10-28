from collections.abc import Iterable
from typing import Any, final

from haiway import freeze

__all__ = [
    "ConstantMemory",
    "VolatileMemory",
    "VolatileAccumulativeMemory",
]


@final
class ConstantMemory[Recalled, Remembered]:
    def __init__(
        self,
        item: Recalled,
        /,
    ) -> None:
        self._item: Recalled = item

        freeze(self)

    async def recall(
        self,
        **extra: Any,
    ) -> Recalled:
        return self._item

    async def remember(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None:
        pass  # ignore


@final
class VolatileMemory[Item]:
    def __init__(
        self,
        item: Item,
        /,
    ) -> None:
        self._item: Item = item

    async def recall(
        self,
        **extra: Any,
    ) -> Item:
        return self._item

    async def remember(
        self,
        *items: Item,
        **extra: Any,
    ) -> None:
        if items:
            self._item = items[-1]


@final
class VolatileAccumulativeMemory[Item]:
    def __init__(
        self,
        items: Iterable[Item] | None = None,
        /,
        limit: int | None = None,
    ) -> None:
        self._items: list[Item] = list(items) if items else []
        self._limit: int = limit or 0

    async def recall(
        self,
        **extra: Any,
    ) -> list[Item]:
        return self._items

    async def remember(
        self,
        *items: Item,
        **extra: Any,
    ) -> None:
        if not items:
            return  # nothing to do

        self._items.extend(items)

        if self._limit > 0:
            self._items = self._items[-self._limit :]
