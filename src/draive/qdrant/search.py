from collections.abc import Sequence
from typing import Any, Literal, overload
from uuid import UUID

from haiway import AttributeRequirement, as_list
from qdrant_client.conversions.common_types import ScoredPoint

from draive.parameters import DataModel
from draive.qdrant.filters import prepare_filter
from draive.qdrant.session import QdrantSession
from draive.qdrant.types import QdrantResult

__all__ = ("QdrantSearchMixin",)


class QdrantSearchMixin(QdrantSession):
    @overload
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        include_vector: Literal[False] = False,
        **extra: Any,
    ) -> Sequence[Model]: ...

    @overload
    async def search[Model: DataModel](
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
    async def search[Model: DataModel](
        self,
        model: type[Model],
        /,
        *,
        query_vector: Sequence[float],
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int = 8,
        include_vector: bool,
        **extra: Any,
    ) -> Sequence[QdrantResult[Model]] | Sequence[Model]: ...

    async def search[Model: DataModel](
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
        results: list[ScoredPoint] = await self.client.search(
            collection_name=model.__name__,
            query_filter=prepare_filter(requirements=requirements),
            query_vector=as_list(query_vector),
            score_threshold=score_threshold,
            limit=limit,
            with_payload=True,
            with_vectors=include_vector,
            **extra,
        )

        if include_vector:
            return tuple(
                _qdrant_result(
                    model,
                    data=result,
                )
                for result in results
            )

        else:
            return tuple(
                model(**result.payload) for result in results if result.payload is not None
            )


def _qdrant_result[Content: DataModel](
    content: type[Content],
    /,
    data: ScoredPoint,
) -> QdrantResult[Content]:
    if data.payload is None:
        raise ValueError("Missing qdrant data payload")

    identifier: UUID
    if isinstance(data.id, int):
        identifier = UUID(int=data.id)

    else:
        assert isinstance(data.id, str)  # nosec: B101
        identifier = UUID(hex=data.id)

    return QdrantResult(
        identifier=identifier,
        vector=_flat_vector(data.vector),
        score=data.score,
        content=content.from_mapping(data.payload),
    )


def _flat_vector(
    vector: Any,
    /,
) -> Sequence[float]:
    match vector:
        case [*vector]:
            assert all(isinstance(element, float) for element in vector)  # nosec: B101
            return tuple(vector)

        case None:
            raise ValueError("Missing qdrant data vector")

        case _:
            raise ValueError("Unsupported qdrant data vector")
