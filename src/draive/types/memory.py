from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "Memory",
    "BasicMemory",
]


class Memory[Recalled, Remembered](ABC):
    @abstractmethod
    async def recall(
        self,
        **extra: Any,
    ) -> Recalled: ...

    @abstractmethod
    async def remember(
        self,
        *items: Remembered,
        **extra: Any,
    ) -> None: ...


type BasicMemory[Value] = Memory[Value, Value]
