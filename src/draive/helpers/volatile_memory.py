from types import TracebackType
from typing import final

__all__ = [
    "VolatileMemory",
]


@final
class VolatileMemory[Item]:
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # noop
