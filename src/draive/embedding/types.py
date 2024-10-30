from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from haiway import State

__all__ = [
    "Embedded",
    "ValueEmbedder",
]


class Embedded[Value](State):
    value: Value
    vector: list[float]


@runtime_checkable
class ValueEmbedder[Value](Protocol):
    async def __call__(
        self,
        values: Sequence[Value],
        **extra: Any,
    ) -> list[Embedded[Value]]: ...
