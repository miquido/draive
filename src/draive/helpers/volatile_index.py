from collections.abc import Callable, Sequence
from typing import Any, cast

from draive.embedding import Embedded, embed_text, embed_texts
from draive.parameters import DataModel, Field, ParameterPath, ParameterRequirement, State
from draive.similarity import mmr_vector_similarity_search, vector_similarity_search

__all__ = [
    "VolatileVectorIndex",
]


class VolatileVectorIndex(State):
    storage: dict[type[Any], list[Embedded[Any]]] = Field(default_factory=dict)

    async def index[Model: DataModel, Value: str](
        self,
        model: type[Model],
        /,
        values: Sequence[Model],
        indexed_value: Callable[[Model], Value] | ParameterPath[Model, Value] | Value,
        **extra: Any,
    ) -> None:
        text_selector: Callable[[Model], Value]
        match indexed_value:
            case Callable() as selector:
                text_selector = selector

            case path:
                assert isinstance(  # nosec: B101
                    path, ParameterPath
                ), "Prepare parameter path by using Self._.path.to.property"
                text_selector = cast(ParameterPath[Model, Value], path).__call__

        embedded_texts: list[Embedded[str]] = await embed_texts(
            [text_selector(value) for value in values],
            **extra,
        )

        embedded_models: list[Embedded[Model]] = [
            Embedded(value=values[index], vector=embedded.vector)
            for index, embedded in enumerate(embedded_texts)
        ]

        if model in self.storage:
            self.storage[model].extend(embedded_models)

        else:
            self.storage[model] = embedded_models

    async def search[Model: DataModel](  # noqa: PLR0913
        self,
        model: type[Model],
        /,
        query: str,
        requirements: ParameterRequirement[Model] | None = None,
        score_threshold: float | None = None,
        limit: int = 10,
        **extra: Any,
    ) -> list[Model]:
        stored: list[Embedded[Model]] | None = self.storage.get(model)
        if not stored:
            return []

        filtered: list[Embedded[Model]]
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

        embedded_query: Embedded[str] = await embed_text(
            query,
            **extra,
        )

        matching: list[Embedded[Model]] = [
            filtered[index]
            for index in vector_similarity_search(
                query_vector=embedded_query.vector,
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
                query_vector=embedded_query.vector,
                values_vectors=[element.vector for element in matching],
                limit=limit,
            )
        ]
