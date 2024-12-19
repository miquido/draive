from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from haiway import State

from draive.parameters import DataModel

__all__ = [
    "Embedded",
    "ValueEmbedder",
]


class Embedded[Value: DataModel | State | str | bytes](State):
    value: Value
    vector: Sequence[float]


@runtime_checkable
class ValueEmbedder[Value: DataModel | State | str | bytes](Protocol):
    async def __call__(
        self,
        values: Sequence[Value],
        **extra: Any,
    ) -> Sequence[Embedded[Value]]: ...
