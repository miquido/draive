from typing import Any, Protocol, Self, final, overload, runtime_checkable

from haiway import State, ctx, statemethod

__all__ = (
    "Memory",
    "MemoryMaintaining",
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


@runtime_checkable
class MemoryMaintaining(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> None: ...


async def _remembering_none(
    *items: Any,
    **extra: Any,
) -> None:
    pass  # noop


async def _maintaining_noop(
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
        async def recalling(
            **extra: Any,
        ) -> Recalled:
            return recalled

        return cls(
            recalling=recalling,
            remembering=_remembering_none,
        )

    @overload
    @classmethod
    async def recall(
        cls,
        **extra: Any,
    ) -> Recalled: ...

    @overload
    async def recall(
        self,
        **extra: Any,
    ) -> Recalled: ...

    @statemethod
    async def recall(
        self,
        **extra: Any,
    ) -> Recalled:
        ctx.record_info(
            event="memory.recall",
        )
        return await self.recalling(**extra)

    @overload
    @classmethod
    async def remember(
        cls,
        *items: Remembered,
        **extra: Any,
    ) -> None: ...

    @overload
    async def remember(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def remember(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None:
        ctx.record_info(
            event="memory.remember",
            attributes={"items": len(items)},
        )
        await self.remembering(*items, **extra)

    @overload
    @classmethod
    async def maintenance(
        cls,
        **extra: Any,
    ) -> None: ...

    @overload
    async def maintenance(
        self,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def maintenance(
        self,
        **extra: Any,
    ) -> None:
        ctx.record_info(
            event="memory.maintenance",
        )
        await self.maintaining(**extra)

    recalling: MemoryRecalling[Recalled]
    remembering: MemoryRemembering[Remembered]
    maintaining: MemoryMaintaining = _maintaining_noop
