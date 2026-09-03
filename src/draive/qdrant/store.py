import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, overload
from uuid import uuid4

from haiway import (
    AttributePath,
    AttributeRequirement,
    Paginated,
    Pagination,
    State,
    as_list,
)
from qdrant_client.conversions.common_types import PointId
from qdrant_client.models import (
    CollectionsResponse,
    Datatype,
    Distance,
    Filter,
    FilterSelector,
    Memory,
    PayloadSchemaType,
    PointStruct,
    Record,
    VectorParams,
)

from draive.embedding import Embedded
from draive.qdrant.filters import prepare_filter
from draive.qdrant.session import QdrantSession
from draive.qdrant.utils import qdrant_arguments, qdrant_operation, qdrant_vector
from draive.utils.attributes import attribute_path_segments

__all__ = ("QdrantStoreMixin",)


class QdrantStoreMixin(QdrantSession):
    async def collections(self) -> Sequence[str]:
        collections: CollectionsResponse
        with qdrant_operation("collection listing"):
            collections = await self.client.get_collections()

        return tuple(collection.name for collection in collections.collections)

    async def create_collection[Model: State](
        self,
        model: type[Model],
        /,
        *,
        vector_size: int,
        vector_type: Literal["float32", "float16", "uint8", "turbo4"] | None = None,
        distance: Literal[
            "Cosine",
            "Euclid",
            "Dot",
            "Manhattan",
        ] = "Cosine",
        in_ram: bool,
        skip_existing: bool,
        **extra: Any,
    ) -> bool:
        # verified eagerly - unrecognized arguments are an error even when skipping
        arguments: Mapping[str, Any] = qdrant_arguments(self.client.create_collection, **extra)

        with qdrant_operation("collection creating", model.__name__):
            if skip_existing and await self.client.collection_exists(
                collection_name=model.__name__
            ):
                return False

            return await self.client.create_collection(
                collection_name=model.__name__,
                vectors_config=VectorParams(
                    size=vector_size,
                    datatype=Datatype(vector_type) if vector_type is not None else None,
                    distance=Distance(distance),
                    # `on_disk` is deprecated in favor of `memory` - `cached` and `cold`
                    # are its defaults for a disabled and enabled flag respectively,
                    # `pinned` is not supported for dense vector storage
                    memory=Memory.CACHED if in_ram else Memory.COLD,
                ),
                on_disk_payload=True,
                **arguments,
            )

    async def create_payload_index[Model: State, Attribute](
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
        # verified eagerly - unrecognized arguments are an error even when skipping
        arguments: Mapping[str, Any] = qdrant_arguments(self.client.create_payload_index, **extra)

        with qdrant_operation("collection index creating", model.__name__):
            if not await self.client.collection_exists(collection_name=model.__name__):
                return False

            await self.client.create_payload_index(
                collection_name=model.__name__,
                field_name=".".join(attribute_path_segments(path)),
                field_schema=PayloadSchemaType(index_type),
                wait=True,
                **arguments,
            )

        return True

    async def delete_collection[Model: State](
        self,
        model: type[Model],
        /,
    ) -> None:
        with qdrant_operation("collection deleting", model.__name__):
            await self.client.delete_collection(collection_name=model.__name__)

    @overload
    async def fetch[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        pagination: Pagination | None = None,
        include_vector: Literal[False] = False,
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
        include_vector: bool,
        **extra: Any,
    ) -> Paginated[Embedded[Model]] | Paginated[Model]: ...

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
        pagination = pagination if pagination is not None else Pagination.of(limit=32)
        assert isinstance(pagination.token, PointId | None)  # nosec: B101
        records: Sequence[Record]
        continuation_token: Any | None
        arguments: Mapping[str, Any] = qdrant_arguments(self.client.scroll, **extra)
        with qdrant_operation("fetching", model.__name__):
            records, continuation_token = await self.client.scroll(
                collection_name=model.__name__,
                scroll_filter=prepare_filter(requirements=requirements),
                limit=pagination.limit,
                offset=pagination.token,
                with_payload=True,
                with_vectors=include_vector,
                **arguments,
            )

        if include_vector:
            return Paginated[Embedded[model]](
                items=[
                    Embedded[model](
                        value=model.from_mapping(record.payload),
                        # we are using only a single vector
                        vector=qdrant_vector(record.vector),
                    )
                    for record in records
                    if record.payload is not None
                ],
                pagination=pagination.with_token(continuation_token),
            )

        else:
            return Paginated[model](
                items=[
                    model.from_mapping(record.payload)
                    for record in records
                    if record.payload is not None
                ],
                pagination=pagination.with_token(continuation_token),
            )

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
        partial_store: Callable[[], None] = self._partial_store(
            model,
            objects=objects,
            batch_size=batch_size,
            max_retries=max_retries,
            parallel_tasks=parallel_tasks,
            **extra,
        )
        with qdrant_operation("storing", model.__name__):
            await asyncio.to_thread(partial_store)

    def _partial_store[Model: State](
        self,
        model: type[Model],
        /,
        *,
        objects: Iterable[Embedded[Model]],
        batch_size: int = 64,
        max_retries: int = 3,
        parallel_tasks: int = 1,
        **extra: Any,
    ) -> Callable[[], None]:
        # verified eagerly, the closure below runs within a worker thread
        arguments: Mapping[str, Any] = qdrant_arguments(self.client.upload_points, **extra)

        def store() -> None:
            # upload_points is a bit weird - spawns multiple threads/processes
            # while blocking current one, we need to offload it to executor
            self.client.upload_points(
                collection_name=model.__name__,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        # qdrant payloads have to be json native - python values
                        # like UUID/datetime are rejected by the grpc uploader
                        payload=dict(element.value.to_basic_object()),
                        vector=as_list(element.vector),
                    )
                    for element in objects
                ],
                batch_size=batch_size,
                max_retries=max_retries,
                parallel=parallel_tasks,
                method="spawn",
                wait=True,
                **arguments,
            )

        return store

    async def delete[Model: State](
        self,
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None,
        **extra: Any,
    ) -> None:
        arguments: Mapping[str, Any] = qdrant_arguments(self.client.delete, **extra)
        with qdrant_operation("deleting", model.__name__):
            await self.client.delete(
                collection_name=model.__name__,
                points_selector=FilterSelector(
                    filter=prepare_filter(
                        requirements=requirements,
                        default=Filter(),
                    ),
                ),
                wait=True,
                **arguments,
            )
