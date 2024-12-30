from collections.abc import Sequence
from typing import Any, Protocol, Self, final, runtime_checkable

from haiway import State

__all__ = [
    "Memory",
]


@runtime_checkable
class MemoryRecall[Recalled](Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Recalled: ...


@runtime_checkable
class MemoryRemember[Remembered](Protocol):
    async def __call__(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None: ...


@final
class Memory[Recalled, Remembered](State):
    @classmethod
    def constant(
        cls,
        recalled: Recalled,
    ) -> Self:
        async def recall(
            **extra: Any,
        ) -> Recalled:
            return recalled

        async def remember(
            *items: Remembered,
            **extra: Any,
        ) -> None:
            pass  # noop

        return cls(
            recall=recall,
            remember=remember,
        )

    @classmethod
    def volatile[Item](
        cls,
        /,
        *,
        initial: Item,
        limit: int | None = None,
    ) -> "Memory[Item, Item]":
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

    @classmethod
    def accumulative_volatile[Item](
        cls,
        /,
        *,
        initial: Sequence[Item] | None = None,
        limit: int | None = None,
    ) -> "Memory[Sequence[Item], Item]":
        storage: list[Item] = list(initial) if initial else []
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

            storage.extend(items)

            if limit > 0:
                storage = storage[-limit:]

        return Memory[Sequence[Item], Item](
            recall=recall,
            remember=remember,
        )

    recall: MemoryRecall[Recalled]
    remember: MemoryRemember[Remembered]
