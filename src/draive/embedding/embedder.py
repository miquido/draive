from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from draive.embedding.embedded import Embedded

__all__ = [
    "ValueEmbedder",
]


@runtime_checkable
class ValueEmbedder[Value](Protocol):
    async def __call__(
        self,
        values: Sequence[Value],
        **extra: Any,
    ) -> list[Embedded[Value]]: ...
