from abc import ABC, abstractmethod

__all__ = [
    "Memory",
]


class Memory[Recalled, Remembered](ABC):
    @abstractmethod
    async def recall(self) -> Recalled: ...

    @abstractmethod
    async def remember(
        self,
        *items: Remembered,
    ) -> None: ...
