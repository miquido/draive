from collections.abc import Iterable, Sequence
from typing import Any, Literal, overload

from haiway import AttributePath, AttributeRequirement, Paginated, Pagination, State, statemethod

from draive.embedding import Embedded
from draive.qdrant.types import (
    QdrantCollectionCreating,
    QdrantCollectionDeleting,
    QdrantCollectionIndexCreating,
    QdrantCollectionListing,
    QdrantDeleting,
    QdrantFetching,
    QdrantResult,
    QdrantSearching,
    QdrantStoring,
)

__all__ = ("Qdrant",)


class Qdrant(State):
    @overload
    @classmethod
    async def collections(cls) -> Sequence[str]: ...

    @overload
    async def collections(self) -> Sequence[str]: ...

    @statemethod
    async def collections(self) -> Sequence[str]:
        return await self.collection_listing()

    @overload
    @classmethod
    async def create_collection[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        vector_size: int,
        vector_type: Literal["float32", "float16", "uint8"] | None = None,
        distance: Literal[
            "Cosine",
            "Euclid",
            "Dot",
            "Manhattan",
        ] = "Cosine",
        in_ram: bool = False,
        skip_existing: bool = True,
        **extra: Any,
    ) -> bool: ...

    @overload
    async def create_collection[Model: State](
        self,
        model: type[Model],
        /,
        *,
        vector_size: int,
        vector_type: Literal["float32", "float16", "uint8"] | None = None,
        distance: Literal[
            "Cosine",
            "Euclid",
            "Dot",
            "Manhattan",
        ] = "Cosine",
        in_ram: bool = False,
        skip_existing: bool = True,
        **extra: Any,
    ) -> bool: ...

    @statemethod
    async def create_collection[Model: State](
        self,
        model: type[Model],
        /,
        *,
        vector_size: int,
        vector_type: Literal["float32", "float16", "uint8"] | None = None,
        distance: Literal[
            "Cosine",
            "Euclid",
            "Dot",
            "Manhattan",
        ] = "Cosine",
        in_ram: bool = False,
        skip_existing: bool = True,
        **extra: Any,
    ) -> bool:
        return await self.collection_creating(
            model,
            vector_size=vector_size,
            vector_type=vector_type,
            distance=distance,
            in_ram=in_ram,
            skip_existing=skip_existing,
            **extra,
        )

    @overload
    @classmethod
    async def delete_collection[Model: State](
        cls,
        model: type[Model],
        /,
    ) -> None: ...

    @overload
    async def delete_collection[Model: State](
        self,
        model: type[Model],
        /,
    ) -> None: ...

    @statemethod
    async def delete_collection[Model: State](
        self,
        model: type[Model],
        /,
    ) -> None:
        return await self.collection_deleting(model)

    @overload
    @classmethod
    async def create_index[Model: State, Attribute](
        cls,
        model: type[Model],
        /,
        *,
        path: AttributePath[Model, Attribute] | Attribute,
        index_type: Literal[
            "keyword",
            "integer",
            "float",
            "geo",
            "text",
            "bool",
            "datetime",
            "uuid",
        ],
        **extra: Any,
    ) -> bool: ...

    @overload
    async def create_index[Model: State, Attribute](
        self,
        model: type[Model],
        /,
        *,
        path: AttributePath[Model, Attribute] | Attribute,
        index_type: Literal[
            "keyword",
            "integer",
            "float",
            "geo",
            "text",
            "bool",
            "datetime",
            "uuid",
        ],
        **extra: Any,
    ) -> bool: ...

    @statemethod
    async def create_index[Model: State, Attribute](
        self,
        model: type[Model],
        /,
        *,
        path: AttributePath[Model, Attribute] | Attribute,
        index_type: Literal[
            "keyword",
            "integer",
            "float",
            "geo",
            "text",
            "bool",
            "datetime",
            "uuid",
        ],
        **extra: Any,
    ) -> bool:
        return await self.collection_index_creating(
            model,
            path=path,
            index_type=index_type,
            **extra,
        )

    @overload
    @classmethod
    async def fetch[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        pagination: Pagination | None = None,
        include_vector: Literal[True],
        **extra: Any,
    ) -> Paginated[Embedded[Model]]: ...

    @overload
    @classmethod
    async def fetch[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[Model]: ...

    @overload
    async def fetch[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        pagination: Pagination | None = None,
        include_vector: Literal[True],
        **extra: Any,
    ) -> Paginated[Embedded[Model]]: ...

    @overload
    async def fetch[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[Model]: ...

    @statemethod
    async def fetch[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        pagination: Pagination | None = None,
        include_vector: bool = False,
        **extra: Any,
    ) -> Paginated[Embedded[Model]] | Paginated[Model]:
        return await self.fetching(
            model,
            requirements=requirements,
            pagination=pagination,
            include_vector=include_vector,
            **extra,
        )

    @overload
    @classmethod
    async def search[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        include_vector: Literal[True],
        **extra: Any,
    ) -> Sequence[QdrantResult[Model]]: ...

    @overload
    @classmethod
    async def search[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    async def search[Model: State](
        self,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        include_vector: Literal[True],
        **extra: Any,
    ) -> Sequence[QdrantResult[Model]]: ...

    @overload
    async def search[Model: State](
        self,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @statemethod
    async def search[Model: State](
        self,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        include_vector: bool = False,
        **extra: Any,
    ) -> Sequence[QdrantResult[Model]] | Sequence[Model]:
        return await self.searching(
            model,
            query_vector=query_vector,
            score_threshold=score_threshold,
            requirements=requirements,
            limit=limit,
            include_vector=include_vector,
            **extra,
        )

    @overload
    @classmethod
    async def store[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        objects: Iterable[Embedded[Model]],
        batch_size: int = 64,
        max_retries: int = 3,
        parallel_tasks: int = 1,
        **extra: Any,
    ) -> None: ...

    @overload
    async def store[Model: State](
        self,
        model: type[Model],
        /,
        *,
        objects: Iterable[Embedded[Model]],
        batch_size: int = 64,
        max_retries: int = 3,
        parallel_tasks: int = 1,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def store[Model: State](
        self,
        model: type[Model],
        /,
        *,
        objects: Iterable[Embedded[Model]],
        batch_size: int = 64,
        max_retries: int = 3,
        parallel_tasks: int = 1,
        **extra: Any,
    ) -> None:
        return await self.storing(
            model,
            objects=objects,
            batch_size=batch_size,
            max_retries=max_retries,
            parallel_tasks=parallel_tasks,
            **extra,
        )

    @overload
    @classmethod
    async def delete[Model: State](
        cls,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None: ...

    @overload
    async def delete[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def delete[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        return await self.deleting(
            model,
            requirements=requirements,
            **extra,
        )

    collection_listing: QdrantCollectionListing
    collection_creating: QdrantCollectionCreating
    collection_deleting: QdrantCollectionDeleting
    collection_index_creating: QdrantCollectionIndexCreating
    fetching: QdrantFetching
    searching: QdrantSearching
    storing: QdrantStoring
    deleting: QdrantDeleting
