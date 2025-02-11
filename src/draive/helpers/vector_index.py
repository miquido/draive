from asyncio import Lock
from collections.abc import Callable, MutableMapping, MutableSequence, Sequence
from typing import Any, Protocol, Self, cast, final, runtime_checkable

from haiway import AttributePath, AttributeRequirement, State

from draive.embedding import Embedded, embed_text, embed_texts
from draive.parameters import DataModel
from draive.similarity import mmr_vector_similarity_search, vector_similarity_search

__all__ = [
    "VectorIndex",
    "VectorIndexing",
    "VectorSearching",
]


@runtime_checkable
class VectorIndexing(Protocol):
    async def __call__[Model: DataModel, Value: str](
        self,
        model: type[Model],
        /,
        values: Sequence[Model],
        indexed_value: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class VectorSearching(Protocol):
    async def __call__[Model: DataModel](
        self,
        model: type[Model],
        /,
        query: Sequence[float] | str | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        score_threshold: float | None = None,
        limit: int = 10,
        **extra: Any,
    ) -> Sequence[Model]: ...


@final
class VectorIndex(State):
    @classmethod
    def volatile(cls) -> Self:  # noqa: C901
        lock: Lock = Lock()
        storage: MutableMapping[type[Any], MutableSequence[Embedded[Any]]] = {}

        async def index[Model: DataModel, Value: str](
            model: type[Model],
            /,
            values: Sequence[Model],
            indexed_value: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
            **extra: Any,
        ) -> None:
            async with lock:
                text_selector: Callable[[Model], Value]
                match indexed_value:
                    case Callable() as selector:
                        text_selector = selector

                    case path:
                        assert isinstance(  # nosec: B101
                            path, AttributePath
                        ), "Prepare parameter path by using Self._.path.to.property"
                        text_selector = cast(AttributePath[Model, Value], path).__call__

                embedded_texts: Sequence[Embedded[str]] = await embed_texts(
                    [text_selector(value) for value in values],
                    **extra,
                )

                embedded_models: list[Embedded[Model]] = [
                    Embedded(value=values[index], vector=embedded.vector)
                    for index, embedded in enumerate(embedded_texts)
                ]

                if model in storage:
                    storage[model].extend(embedded_models)

                else:
                    storage[model] = embedded_models

        async def search[Model: DataModel](
            model: type[Model],
            /,
            query: Sequence[float] | str | None = None,
            requirements: AttributeRequirement[Model] | None = None,
            score_threshold: float | None = None,
            limit: int = 10,
            **extra: Any,
        ) -> Sequence[Model]:
            async with lock:
                stored: Sequence[Embedded[Model]] | None = storage.get(model)
                if not stored:
                    return []

                filtered: Sequence[Embedded[Model]]
                if requirements:
                    filtered = [
                        element
                        for element in stored
                        if requirements.check(
                            element.value,
                            raise_exception=False,
                        )
                    ]

                else:
                    filtered = stored

                if not filtered:
                    return []

                query_vector: Sequence[float]
                match query:
                    case None:
                        if limit:
                            return [embedded.value for embedded in filtered[:limit]]

                        else:
                            return [embedded.value for embedded in filtered]

                    case str() as text:
                        embedded_text: Embedded[str] = await embed_text(
                            text,
                            **extra,
                        )
                        query_vector = embedded_text.vector

                    case vector:
                        query_vector = vector

                matching: Sequence[Embedded[Model]] = [
                    filtered[index]
                    for index in vector_similarity_search(
                        query_vector=query_vector,
                        values_vectors=[element.vector for element in filtered],
                        score_threshold=score_threshold,
                        limit=limit * 8,  # feed MMR with more results
                    )
                ]

                if not matching:
                    return []

                return [
                    matching[index].value
                    for index in mmr_vector_similarity_search(
                        query_vector=query_vector,
                        values_vectors=[element.vector for element in matching],
                        limit=limit,
                    )
                ]

        return cls(
            index=index,
            search=search,
        )

    index: VectorIndexing
    search: VectorSearching
