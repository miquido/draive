from collections.abc import Callable, Collection, Sequence
from typing import Any, Protocol, runtime_checkable

from haiway import (
    META_EMPTY,
    AttributePath,
    AttributeRequirement,
    Meta,
    State,
)

from draive.multimodal import TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent

__all__ = (
    "Embedded",
    "ValueEmbedding",
    "VectorDeleting",
    "VectorIndexing",
    "VectorSearching",
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


@runtime_checkable
class VectorIndexing(Protocol):
    async def __call__[Model: DataModel, Value: ResourceContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class VectorSearching(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str | None,
        score_threshold: float | None,
        requirements: AttributeRequirement[Model] | None,
        limit: int | None,
        **extra: Any,
    ) -> Sequence[Model]: ...


@runtime_checkable
class VectorDeleting(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None,
        **extra: Any,
    ) -> None: ...
