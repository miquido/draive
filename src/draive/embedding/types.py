from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from haiway import META_EMPTY, Meta, State

from draive.parameters import DataModel

__all__ = (
    "Embedded",
    "ValueEmbedding",
)


class Embedded[Value: DataModel | State | str | bytes](State):
    value: Value
    vector: Sequence[float]
    meta: Meta = META_EMPTY


@runtime_checkable
class ValueEmbedding[Value: DataModel | State | str | bytes, Data: str | bytes](Protocol):
    async def __call__(
        self,
        values: Sequence[Value] | Sequence[Data],
        /,
        attribute: Callable[[Value], Data] | None = None,
        **extra: Any,
    ) -> Sequence[Embedded[Value]] | Sequence[Embedded[Data]]: ...
