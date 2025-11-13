from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from haiway import AttributePath, AttributeRequirement, State

from draive.embedding import Embedded
from draive.parameters import DataModel

__all__ = (
    "QdrantCollectionCreating",
    "QdrantCollectionDeleting",
    "QdrantCollectionIndexCreating",
    "QdrantCollectionListing",
    "QdrantDeleting",
    "QdrantException",
    "QdrantFetching",
    "QdrantPaginationResult",
    "QdrantPaginationToken",
    "QdrantResult",
    "QdrantSearching",
    "QdrantStoring",
)


class QdrantException(Exception):
    pass


class QdrantResult[Content: DataModel](State):
    identifier: UUID
    vector: Sequence[float]
    score: float
    content: Content


class QdrantPaginationToken(State):
    next_id: Any


class QdrantPaginationResult[Result](State):
    results: Sequence[Result]
    continuation_token: QdrantPaginationToken | None


@runtime_checkable
class QdrantCollectionListing(Protocol):
    async def __call__(self) -> Sequence[str]: ...


@runtime_checkable
class QdrantCollectionCreating(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        vector_size: int,
        vector_type: Literal["float32", "float16", "uint8"] | None,
        distance: Literal[
            "Cosine",
            "Euclid",
            "Dot",
            "Manhattan",
        ],
        in_ram: bool,
        skip_existing: bool,
        **extra: Any,
    ) -> bool: ...


@runtime_checkable
class QdrantCollectionIndexCreating(Protocol):
    async def __call__[Model: DataModel, Attribute](
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


@runtime_checkable
class QdrantCollectionDeleting(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
    ) -> None: ...


@runtime_checkable
class QdrantFetching(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None,
        continuation: QdrantPaginationToken | None,
        limit: int,
        include_vector: bool,
        **extra: Any,
    ) -> QdrantPaginationResult[Embedded[Model]] | QdrantPaginationResult[Model]: ...


@runtime_checkable
class QdrantSearching(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        requirements: AttributeRequirement[Model] | None,
        score_threshold: float | None,
        limit: int,
        include_vector: bool,
        **extra: Any,
    ) -> Sequence[QdrantResult[Model]] | Sequence[Model]: ...


@runtime_checkable
class QdrantStoring(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        objects: Iterable[Embedded[Model]],
        batch_size: int,
        max_retries: int,
        parallel_tasks: int,
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class QdrantDeleting(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None,
        **extra: Any,
    ) -> None: ...
