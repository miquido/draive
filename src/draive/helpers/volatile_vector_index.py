from asyncio import Lock
from collections.abc import Callable, Collection, MutableMapping, MutableSequence, Sequence
from typing import Any, cast

from haiway import AttributePath, AttributeRequirement

from draive.embedding import (
    Embedded,
    ImageEmbedding,
    TextEmbedding,
    mmr_vector_similarity_search,
    vector_similarity_search,
)
from draive.multimodal import TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent
from draive.utils import VectorIndex

__all__ = ("VolatileVectorIndex",)


def VolatileVectorIndex() -> VectorIndex:  # noqa: C901, PLR0915
    lock: Lock = Lock()
    storage: MutableMapping[type[Any], MutableSequence[Embedded[Any]]] = {}

    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None:
        async with lock:
            assert isinstance(  # nosec: B101
                attribute, AttributePath | Callable
            ), f"Prepare parameter path by using {model.__name__}._.path.to.property"
            value_selector: Callable[[Model], Value] = cast(Callable[[Model], Value], attribute)

            text_values: list[str] = []
            image_values: list[bytes] = []
            for value in values:
                selected = value_selector(value)
                if isinstance(selected, str):
                    text_values.append(selected)

                elif isinstance(selected, TextContent):
                    text_values.append(selected.text)

                else:
                    assert isinstance(selected, ResourceContent)  # nosec: B101
                    if not selected.mime_type.startswith("image"):
                        raise ValueError(f"{selected.mime_type} embedding is not supported")

                    image_values.append(selected.to_bytes())

            if image_values and text_values:
                raise ValueError("Selected attribute values have to be the same type")

            embedded_values: Sequence[Embedded[str] | Embedded[bytes]]
            if image_values:
                embedded_values = await ImageEmbedding.embed_many(
                    image_values,
                    **extra,
                )

            else:
                embedded_values = await TextEmbedding.embed_many(
                    text_values,
                    **extra,
                )

            embedded_models: list[Embedded[Model]] = [
                Embedded(
                    value=value,
                    vector=embedded.vector,
                )
                for value, embedded in zip(
                    values,
                    embedded_values,
                    strict=True,
                )
            ]

            if model in storage:
                storage[model].extend(embedded_models)

            else:
                storage[model] = embedded_models

    async def search[Model: DataModel](  # noqa: C901
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[Model]:
        assert query is not None or (query is None and score_threshold is None)  # nosec: B101
        stored: Sequence[Embedded[Model]] | None
        async with lock:
            stored = storage.get(model)

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
        if query is None:
            if limit:
                return [embedded.value for embedded in filtered[:limit]]

            return [embedded.value for embedded in filtered]

        if isinstance(query, str):
            embedded_text: Embedded[str] = await TextEmbedding.embed(
                query,
                **extra,
            )
            query_vector = embedded_text.vector

        elif isinstance(query, TextContent):
            embedded_text = await TextEmbedding.embed(
                query.text,
                **extra,
            )
            query_vector = embedded_text.vector

        elif isinstance(query, ResourceContent):
            if not query.mime_type.startswith("image"):
                raise ValueError(f"{query.mime_type} embedding is not supported")

            embedded_image: Embedded[bytes] = await ImageEmbedding.embed(
                query.to_bytes(),
                **extra,
            )
            query_vector = embedded_image.vector

        else:
            assert isinstance(query, Sequence)  # nosec: B101
            query_vector = query  # vector

        matching: Sequence[Embedded[Model]] = [
            filtered[index]
            for index in vector_similarity_search(
                query_vector=query_vector,
                values_vectors=[element.vector for element in filtered],
                score_threshold=score_threshold,
                limit=limit * 8  # feed MMR with more results
                if limit is not None
                else None,
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

    async def delete[Model: DataModel](
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        async with lock:
            if model not in storage:
                return

            if requirements is not None:
                storage[model] = [
                    element
                    for element in storage[model]
                    if not requirements.check(
                        element.value,
                        raise_exception=False,
                    )
                ]

            else:
                del storage[model]

    return VectorIndex(
        indexing=index,
        searching=search,
        deleting=delete,
    )
