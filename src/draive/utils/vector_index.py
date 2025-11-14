from collections.abc import Callable, Collection, Sequence
from typing import Any, Protocol, final, overload, runtime_checkable

from haiway import (
    AttributePath,
    AttributeRequirement,
    State,
    ctx,
    statemethod,
)

from draive.multimodal import TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent

__all__ = ("VectorIndex",)


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


@final
class VectorIndex(State):
    @overload
    @classmethod
    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        cls,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None: ...

    @overload
    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        self,
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None:
        ctx.record_info(
            event="vector_index.index",
            attributes={
                "model": model.__qualname__,
                "values": len(values),
            },
        )
        return await self.indexing(
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
    ) -> Sequence[Model]: ...

    @overload
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    @classmethod
    async def search[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @statemethod
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]:
        ctx.record_info(
            event="vector_index.search",
            attributes={"model": model.__qualname__},
        )
        results: Sequence[Model] = await self.searching(
            model,
            query=query,
            score_threshold=score_threshold,
            requirements=requirements,
            limit=limit,
            **extra,
        )

        return results

    @overload
    @classmethod
    async def delete[Model: DataModel](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None: ...

    @overload
    async def delete[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def delete[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        ctx.record_info(
            event="vector_index.delete",
            attributes={"model": model.__qualname__},
        )
        return await self.deleting(
            model,
            requirements=requirements,
            **extra,
        )

    indexing: VectorIndexing
    searching: VectorSearching
    deleting: VectorDeleting
