from typing import Any, Protocol, Self, final, runtime_checkable

from haiway import State

__all__ = (
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

    recall: MemoryRecalling[Recalled]
    remember: MemoryRemembering[Remembered]
