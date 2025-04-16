from collections.abc import Sequence
from typing import Any

from draive.utils.memory import Memory

__all__ = (
    "AccumulativeVolatileMemory",
    "ConstantMemory",
    "VolatileMemory",
)


def ConstantMemory[Recalled](
    recalled: Recalled,
) -> Memory[Recalled, Any]:
    async def recall(
        **extra: Any,
    ) -> Recalled:
        return recalled

    async def remember(
        *items: Any,
        **extra: Any,
    ) -> None:
        pass  # noop

    return Memory[Recalled, Any](
        recall=recall,
        remember=remember,
    )


def VolatileMemory[Item](
    *,
    initial: Item,
    limit: int | None = None,
) -> Memory[Item, Item]:
    storage: Item = initial
    limit = limit or 0

    async def recall(
        **extra: Any,
    ) -> Item:
        nonlocal storage
        return storage

    async def remember(
        *items: Item,
        **extra: Any,
    ) -> None:
        nonlocal storage
        if not items:
            return  # nothing to do

        storage = items[-1]

    return Memory[Item, Item](
        recall=recall,
        remember=remember,
    )


def AccumulativeVolatileMemory[Item](
    *,
    initial: Sequence[Item] | None = None,
    limit: int | None = None,
) -> Memory[Sequence[Item], Item]:
    storage: Sequence[Item] = tuple(initial) if initial else ()
    limit = limit or 0

    async def recall(
        **extra: Any,
    ) -> Sequence[Item]:
        nonlocal storage
        return storage

    async def remember(
        *items: Item,
        **extra: Any,
    ) -> None:
        nonlocal storage
        if not items:
            return  # nothing to do

        storage = (*storage, *items)

        if limit > 0:
            storage = tuple(storage[-limit:])

    return Memory[Sequence[Item], Item](
        recall=recall,
        remember=remember,
    )
