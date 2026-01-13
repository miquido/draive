import asyncio
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast, overload
from uuid import uuid4

from haiway import (
    AttributePath,
    AttributeRequirement,
    Paginated,
    Pagination,
    State,
    as_dict,
    as_list,
)
from qdrant_client.grpc import PointId
from qdrant_client.models import (
    CollectionsResponse,
    Datatype,
    Distance,
    Filter,
    FilterSelector,
    PayloadSchemaType,
    PointStruct,
    Record,
    VectorParams,
)

from draive.embedding import Embedded
from draive.qdrant.filters import prepare_filter
from draive.qdrant.session import QdrantSession

__all__ = ("QdrantStoreMixin",)


class QdrantStoreMixin(QdrantSession):
    async def collections(self) -> Sequence[str]:
        collections: CollectionsResponse = await self.client.get_collections()
        return tuple(collection.name for collection in collections.collections)

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
        in_ram: bool,
        skip_existing: bool,
        **extra: Any,
    ) -> bool:
        if skip_existing and await self.client.collection_exists(collection_name=model.__name__):
            return False

        return await self.client.create_collection(
            collection_name=model.__name__,
            vectors_config=VectorParams(
                size=vector_size,
                datatype=Datatype(vector_type) if vector_type is not None else None,
                distance=Distance(distance),
                on_disk=not in_ram,
            ),
            on_disk_payload=True,
            **extra,
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
        if not await self.client.collection_exists(collection_name=model.__name__):
            return False

        await self.client.create_payload_index(
            collection_name=model.__name__,
            field_name=str(path),
            field_type=PayloadSchemaType(index_type),
            wait=True,
            **extra,
        )
        return True

    async def delete_collection[Model: State](
        self,
        model: type[Model],
        /,
    ) -> None:
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
        records, continuation_token = await self.client.scroll(
            collection_name=model.__name__,
            scroll_filter=prepare_filter(requirements=requirements),
            limit=pagination.limit,
            offset=pagination.token,
            with_payload=True,
            with_vectors=include_vector,
            **extra,
        )

        if include_vector:
            return Paginated[Embedded[model]](
                items=[
                    Embedded[model](
                        value=model.from_mapping(record.payload),
                        # we are using only a single vector
                        vector=cast(list[float], record.vector),
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
        await asyncio.to_thread(
            self._partial_store(
                model,
                objects=objects,
                batch_size=batch_size,
                max_retries=max_retries,
                parallel_tasks=parallel_tasks,
                **extra,
            ),
        )

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
        def store() -> None:
            # upload_points is a bit weird - spawns multiple threads/processes
            # while blocking current one, we need to offload it to executor
            self.client.upload_points(
                collection_name=model.__name__,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        payload=as_dict(element.value.to_mapping()),
                        vector=as_list(element.vector),
                    )
                    for element in objects
                ],
                batch_size=batch_size,
                max_retries=max_retries,
                parallel=parallel_tasks,
                method="spawn",
                wait=True,
                **extra,
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
        await self.client.delete(
            collection_name=model.__name__,
            points_selector=FilterSelector(
                filter=prepare_filter(
                    requirements=requirements,
                    default=Filter(),
                ),
            ),
            wait=True,
            **extra,
        )
