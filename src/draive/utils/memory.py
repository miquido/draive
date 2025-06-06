from typing import Any, Protocol, final, runtime_checkable

from haiway import State

__all__ = ("Memory", "MemoryRecalling", "MemoryRemembering")


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


@final
class Memory[Recalled, Remembered](State):
    recall: MemoryRecalling[Recalled]
    remember: MemoryRemembering[Remembered]
