from asyncio import Lock
from collections.abc import Callable, Iterable, MutableMapping, MutableSequence, Sequence
from typing import Any, cast

from haiway import AttributePath, AttributeRequirement

from draive.embedding import Embedded, ImageEmbedding, TextEmbedding
from draive.multimodal import MediaContent, MediaData, MediaReference, TextContent
from draive.parameters import DataModel
from draive.similarity import mmr_vector_similarity_search, vector_similarity_search
from draive.utils import VectorIndex

__all__ = ("VolatileVectorIndex",)


def VolatileVectorIndex() -> VectorIndex:  # noqa: C901, PLR0915
    lock: Lock = Lock()
    storage: MutableMapping[type[Any], MutableSequence[Embedded[Any]]] = {}

    async def index[Model: DataModel, Value: MediaContent | TextContent | str](  # noqa: C901, PLR0912
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Iterable[Model],
        **extra: Any,
    ) -> None:
        async with lock:
            value_selector: Callable[[Model], Value]
            match attribute:
                case Callable() as selector:
                    value_selector = selector

                case path:
                    assert isinstance(  # nosec: B101
                        path, AttributePath
                    ), "Prepare parameter path by using Self._.path.to.property"
                    value_selector = cast(AttributePath[Model, Value], path).__call__

            selected_values: list[str | bytes] = []
            for value in values:
                match value_selector(value):
                    case str() as text:
                        selected_values.append(text)

                    case TextContent() as text_content:
                        selected_values.append(text_content.text)

                    case MediaData() as media_data:
                        if media_data.kind != "image":
                            raise ValueError(f"{media_data.kind} embedding is not supported")

                        selected_values.append(media_data.data)

                    case MediaReference():
                        raise ValueError("Media references are not supported")

            embedded_values: Sequence[Embedded[str] | Embedded[bytes]]
            if all(isinstance(value, str) for value in selected_values):
                embedded_values = await TextEmbedding.embed(
                    cast(list[str], selected_values),
                    **extra,
                )

            elif all(value for value in selected_values):
                embedded_values = await ImageEmbedding.embed(
                    cast(list[bytes], selected_values),
                    **extra,
                )

            else:
                raise ValueError("Selected attribute values have to be the same type")

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

    async def search[Model: DataModel](  # noqa: C901, PLR0912
        model: type[Model],
        /,
        *,
        query: Sequence[float] | MediaContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        **extra: Any,
    ) -> Iterable[Model]:
        assert query is not None or (query is None and score_threshold is None)  # nosec: B101
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
                    embedded_text: Embedded[str] = await TextEmbedding.embed(
                        text,
                        **extra,
                    )
                    query_vector = embedded_text.vector

                case TextContent() as text_content:
                    embedded_text: Embedded[str] = await TextEmbedding.embed(
                        text_content.text,
                        **extra,
                    )
                    query_vector = embedded_text.vector

                case MediaData() as media_data:
                    if media_data.kind != "image":
                        raise ValueError(f"{media_data.kind} embedding is not supported")

                    embedded_image: Embedded[bytes] = await ImageEmbedding.embed(
                        media_data.data,
                        **extra,
                    )
                    query_vector = embedded_image.vector

                case MediaReference():
                    raise ValueError("Media references are not supported")

                case vector:
                    query_vector = vector

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
