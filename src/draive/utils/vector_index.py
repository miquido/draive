from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, final, overload, runtime_checkable

from haiway import AttributePath, AttributeRequirement, State, ctx

from draive.multimodal import MediaContent, TextContent
from draive.parameters import DataModel

__all__ = (
    "VectorIndex",
    "VectorIndexing",
    "VectorSearching",
)


@runtime_checkable
class VectorIndexing(Protocol):
    async def __call__[Model: DataModel, Value: MediaContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Iterable[Model],
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class VectorSearching(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | MediaContent | TextContent | str | None,
        score_threshold: float | None,
        requirements: AttributeRequirement[Model] | None,
        limit: int | None,
        **extra: Any,
    ) -> Iterable[Model]: ...


@runtime_checkable
class VectorDeleting(Protocol):
    async def __call__[Model: DataModel, Value: MediaContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None,
        **extra: Any,
    ) -> None: ...


@final
class VectorIndex(State):
    @classmethod
    async def index[Model: DataModel, Value: MediaContent | TextContent | str](
        cls,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Iterable[Model],
        **extra: Any,
    ) -> None:
        return await ctx.state(cls).indexing(
            model,
            attribute=attribute,
            values=values,
            **extra,
        )

    @overload
    @classmethod
    async def search[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Iterable[Model]: ...

    @overload
    @classmethod
    async def search[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | MediaContent | TextContent | str,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Iterable[Model]: ...

    @classmethod
    async def search[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | MediaContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Iterable[Model]:
        return await ctx.state(cls).searching(
            model,
            query=query,
            score_threshold=score_threshold,
            requirements=requirements,
            limit=limit,
            **extra,
        )

    @classmethod
    async def delete[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        return await ctx.state(cls).deleting(
            model,
            requirements=requirements,
            **extra,
        )

    indexing: VectorIndexing
    searching: VectorSearching
    deleting: VectorDeleting
