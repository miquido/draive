from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from draive.types.embedded import Embedded

__all__ = [
    "Embedder",
]


_Embeddable = TypeVar(
    "_Embeddable",
    bound=object,
)


@runtime_checkable
class Embedder(Protocol):
    async def __call__(
        self,
        values: Iterable[_Embeddable],
    ) -> list[Embedded[_Embeddable]]:
        ...
