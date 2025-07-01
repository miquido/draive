from collections.abc import Sequence
from typing import Any, Final, Protocol, Self, final, runtime_checkable

from haiway import State

__all__ = (
    "MEMORY_NONE",
    "Memory",
    "MemoryRecalling",
    "MemoryRemembering",
)


@runtime_checkable
class MemoryRecalling[Recalled](Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Recalled: ...


@runtime_checkable
class MemoryRemembering[Remembered](Protocol):
    async def __call__(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None: ...


async def _recall_none(
    **extra: Any,
) -> Any:
    return ()


async def _remember_none(
    *items: Any,
    **extra: Any,
) -> None:
    pass  # noop


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

        return cls(
            recall=recall,
            remember=_remember_none,
        )

    @classmethod
    def volatile[Item](
        cls,
        *,
        initial: Item,
    ) -> "Memory[Item, Item]":
        storage: Item = initial

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
        *,
        initial: Sequence[Item] | None = None,
        limit: int | None = None,
    ) -> "Memory[Sequence[Item], Item]":
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

    recall: MemoryRecalling[Recalled]
    remember: MemoryRemembering[Remembered]


MEMORY_NONE: Final[Memory[Any, Any]] = Memory[Any, Any](
    recall=_recall_none,
    remember=_remember_none,
)
