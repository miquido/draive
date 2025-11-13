from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast

from haiway import AttributePath, AttributeRequirement

from draive.embedding import Embedded, ImageEmbedding, TextEmbedding, mmr_vector_similarity_search
from draive.multimodal import TextContent
from draive.parameters import DataModel
from draive.qdrant.state import Qdrant
from draive.qdrant.types import QdrantResult
from draive.resources import ResourceContent
from draive.utils import VectorIndex

__all__ = ("QdrantVectorIndex",)


def QdrantVectorIndex() -> VectorIndex:  # noqa: C901
    """VectorIndex that manages text/image embeddings stored in Qdrant."""

    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        model: type[Model],
        /,
        values: Iterable[Model],
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        **extra: Any,
    ) -> None:
        assert isinstance(  # nosec: B101
            attribute, AttributePath | Callable
        ), f"Prepare parameter path by using {model.__name__}._.path.to.property"
        value_selector: Callable[[Model], Value] = cast(Callable[[Model], Value], attribute)

        selected_texts: list[str] = []
        selected_images: list[bytes] = []
        embedding_plan: list[tuple[Model, Literal["text", "image"], int]] = []

        for value in values:
            selected: Value = value_selector(value)
            if isinstance(selected, str):
                index = len(selected_texts)
                selected_texts.append(selected)
                embedding_plan.append((value, "text", index))

            elif isinstance(selected, TextContent):
                index = len(selected_texts)
                selected_texts.append(selected.text)
                embedding_plan.append((value, "text", index))

            else:
                assert isinstance(selected, ResourceContent)  # nosec: B101
                if not selected.mime_type.startswith("image"):
                    raise ValueError(f"{selected.mime_type} embedding is not supported")

                index = len(selected_images)
                selected_images.append(selected.to_bytes())
                embedding_plan.append((value, "image", index))

        text_embeddings: Sequence[Embedded[str]] = (
            await TextEmbedding.embed_many(selected_texts) if selected_texts else ()
        )
        image_embeddings: Sequence[Embedded[bytes]] = (
            await ImageEmbedding.embed_many(selected_images) if selected_images else ()
        )

        await Qdrant.store(
            model,
            objects=[
                Embedded(
                    value=value,
                    vector=(
                        text_embeddings[index].vector
                        if kind == "text"
                        else image_embeddings[index].vector
                    ),
                )
                for value, kind, index in embedding_plan
            ],
            **extra,
        )

    async def search[Model: DataModel](
        model: type[Model],
        /,
        query: Sequence[float] | ResourceContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        rerank: bool = False,
        **extra: Any,
    ) -> Sequence[Model]:
        assert query is not None or (query is None and score_threshold is None)  # nosec: B101

        if query is None:
            return (
                await Qdrant.fetch(
                    model,
                    requirements=requirements,
                    limit=limit or 32,
                    include_vector=False,
                    **extra,
                )
            ).results

        query_vector: Sequence[float]
        if isinstance(query, str):
            embedded_query: Embedded[str] = await TextEmbedding.embed(query)
            query_vector = embedded_query.vector

        elif isinstance(query, TextContent):
            embedded_query: Embedded[str] = await TextEmbedding.embed(query.text)
            query_vector = embedded_query.vector

        elif isinstance(query, ResourceContent):
            if not query.mime_type.startswith("image"):
                raise ValueError(f"{query.mime_type} embedding is not supported")

            embedded_image: Embedded[bytes] = await ImageEmbedding.embed(
                query.to_bytes(),
                **extra,
            )
            query_vector = embedded_image.vector

        else:
            query_vector = query

        search_results: Sequence[QdrantResult[Model]] = await Qdrant.search(
            model,
            query_vector=query_vector,
            score_threshold=score_threshold,
            requirements=requirements,
            limit=limit or 8,
            include_vector=True,
            **extra,
        )

        if not rerank:
            return tuple(result.content for result in search_results)

        matching: Sequence[Embedded[Model]] = tuple(
            Embedded(
                value=result.content,
                vector=result.vector,
            )
            for result in search_results
        )

        return tuple(
            matching[index].value
            for index in mmr_vector_similarity_search(
                query_vector=query_vector,
                values_vectors=[element.vector for element in matching],
                limit=limit,
            )
        )

    async def delete[Model: DataModel](
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        await Qdrant.delete(
            model,
            requirements=requirements,
            **extra,
        )

    return VectorIndex(
        indexing=index,
        searching=search,
        deleting=delete,
    )
