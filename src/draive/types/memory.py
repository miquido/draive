from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Memory",
    "BasicMemory",
]


@runtime_checkable
class Memory[Recalled, Remembered](Protocol):
    async def recall(
        self,
        **extra: Any,
    ) -> Recalled: ...

    async def remember(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None: ...


type BasicMemory[Value] = Memory[Value, Value]
