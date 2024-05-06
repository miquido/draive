from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from draive.embedding.embedded import Embedded

__all__ = [
    "Embedder",
]


@runtime_checkable
class Embedder[Value](Protocol):
    async def __call__(
        self,
        values: Iterable[Value],
        **extra: Any,
    ) -> list[Embedded[Value]]: ...
