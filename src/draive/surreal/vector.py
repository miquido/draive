import re
from base64 import b64decode
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from haiway import AttributePath, AttributeRequirement, State, ctx

from draive.embedding import (
    Embedded,
    ImageEmbedding,
    TextEmbedding,
    VectorIndex,
    mmr_vector_similarity_search,
)
from draive.multimodal import TextContent
from draive.resources import ResourceContent
from draive.surreal.filters import prepare_filter
from draive.surreal.state import Surreal
from draive.surreal.types import SurrealException, SurrealObject, SurrealValue

__all__ = (
    "SurrealDBVectorIndex",
    "SurrealVectorIndex",
)


def SurrealDBVectorIndex(  # noqa: C901, PLR0915
    *,
    mmr_multiplier: int = 8,
    search_effort: int = 40,
    vector_type: str = "F64",
    distance: str = "COSINE",
    efc: int | None = None,
    m: int | None = None,
) -> VectorIndex:
    """Create a SurrealDB-backed implementation of :class:`VectorIndex`.

    Parameters
    ----------
    mmr_multiplier
        Multiplier applied to ``limit`` to determine how many nearest-neighbour
        candidates are fetched before applying Maximal Marginal Relevance (MMR)
        re-ranking.
    search_effort
        HNSW effort passed to SurrealDB's KNN operator during search.
    vector_type
        SurrealDB vector storage type for the HNSW index.
    distance
        SurrealDB distance metric used by the HNSW index.
    efc
        Optional HNSW ``EFC`` value.
    m
        Optional HNSW ``M`` value.

    Returns
    -------
    VectorIndex
        A VectorIndex implementation persisting entries in SurrealDB.
    """
    if mmr_multiplier <= 0:
        raise ValueError("mmr_multiplier has to be greater than 0")

    if search_effort <= 0:
        raise ValueError("search_effort has to be greater than 0")

    vector_type = _surreal_option_token(vector_type, name="vector_type")
    distance = _surreal_option_token(distance, name="distance")
    if efc is not None and efc <= 0:
        raise ValueError("efc has to be greater than 0")

    if m is not None and m <= 0:
        raise ValueError("m has to be greater than 0")

    async def ensure_index(
        model: type[State],
        /,
        *,
        dimensions: int,
    ) -> None:
        options: list[str] = [
            f"TYPE {vector_type}",
            f"DIST {distance}",
        ]
        if efc is not None:
            options.append(f"EFC {efc}")
        if m is not None:
            options.append(f"M {m}")

        await Surreal.execute(
            f"DEFINE INDEX IF NOT EXISTS {model.__name__}_embedding_index "
            f"ON TABLE {model.__name__} FIELDS embedding "
            f"HNSW DIMENSION {dimensions} {' '.join(options)};"
        )

    async def index[Model: State, Value: ResourceContent | TextContent | str](
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None:
        if not isinstance(attribute, AttributePath | Callable):
            raise TypeError(f"Prepare parameter path by using {model.__name__}._.path.to.property")

        value_selector: Callable[[Model], Value] = cast(Callable[[Model], Value], attribute)

        indexed_values: Sequence[Model] = tuple(values)
        if not indexed_values:
            return

        selected_values: list[str | bytes] = [
            _embedding_input(value_selector(value)) for value in indexed_values
        ]

        embedded_values: Sequence[Embedded[str] | Embedded[bytes]]
        if all(isinstance(value, str) for value in selected_values):
            embedded_values = await TextEmbedding.embed_many(
                cast(Sequence[str], selected_values),
                **extra,
            )

        elif all(isinstance(value, bytes) for value in selected_values):
            embedded_values = await ImageEmbedding.embed_many(
                cast(Sequence[bytes], selected_values),
                **extra,
            )

        else:
            raise ValueError("Selected attribute values have to be the same type")

        await ensure_index(
            model,
            dimensions=len(embedded_values[0].vector),
        )

        created_timestamp: datetime = datetime.now(UTC)
        for idx, (value, embedded) in enumerate(zip(indexed_values, embedded_values, strict=True)):
            await Surreal.execute(
                f"""
                CREATE {model.__name__} SET
                    content = $content,
                    embedding = $embedding,
                    created = $created;
                """,
                content=cast(Any, value.to_mapping()),
                embedding=list(embedded.vector),
                created=created_timestamp + timedelta(microseconds=idx),
            )

        ctx.log_info("Vector index update completed.")

    async def search[Model: State](  # noqa: C901, PLR0912
        model: type[Model],
        /,
        *,
        query: Sequence[float] | ResourceContent | TextContent | str | None = None,
        score_threshold: float | None = None,
        requirements: AttributeRequirement[Model] | None = None,
        limit: int | None = None,
        rerank: bool = False,
        **extra: Any,
    ) -> Sequence[Model]:
        if query is None and score_threshold is not None:
            raise ValueError("score_threshold requires a query")
        if score_threshold is not None:
            if distance != "COSINE":
                raise ValueError(
                    "score_threshold is only supported when distance='COSINE' because "
                    "thresholds are applied to cosine similarity"
                )

            if score_threshold < -1.0 or score_threshold > 1.0:
                raise ValueError("COSINE score_threshold has to be within [-1.0, 1.0]")

        scoped_requirements: AttributeRequirement[Model] | None = _content_scoped_requirements(
            requirements
        )
        filter_clause, filter_variables = prepare_filter(scoped_requirements)
        where_clause: str = f" WHERE {filter_clause}" if filter_clause else ""

        result_limit: int = limit if limit is not None else 8
        if query is None:
            rows: Sequence[SurrealObject] = await Surreal.execute(
                f"""
                SELECT
                    content,
                    created,
                    id
                FROM
                    {model.__name__}
                {where_clause}
                ORDER BY
                    created DESC,
                    id DESC
                LIMIT
                    $limit;
                """,  # nosec: B608
                **cast(Any, filter_variables),
                limit=result_limit,
            )

            return tuple(
                model.from_mapping(cast(Mapping[str, Any], record["content"])) for record in rows
            )

        query_vector: Sequence[float]
        if isinstance(query, str):
            embedded_query: Embedded[str] = await TextEmbedding.embed(query, **extra)
            query_vector = embedded_query.vector

        elif isinstance(query, TextContent):
            embedded_query = await TextEmbedding.embed(query.text, **extra)
            query_vector = embedded_query.vector

        elif isinstance(query, ResourceContent):
            if query.mime_type.startswith("image"):
                embedded_image: Embedded[bytes] = await ImageEmbedding.embed(
                    query.to_bytes(),
                    **extra,
                )
                query_vector = embedded_image.vector

            elif query.mime_type.startswith("text"):
                embedded_query = await TextEmbedding.embed(
                    b64decode(query.data).decode(),
                    **extra,
                )
                query_vector = embedded_query.vector

            else:
                raise ValueError(f"{query.mime_type} embedding is not supported")

        else:
            query_vector = query

        candidate_limit: int = result_limit * mmr_multiplier if rerank else result_limit
        query_where: str
        if filter_clause:
            query_where = (
                f" WHERE ({filter_clause}) "
                f"AND embedding <|{candidate_limit},{search_effort}|> $query"
            )

        else:
            query_where = f" WHERE embedding <|{candidate_limit},{search_effort}|> $query"

        rows = await Surreal.execute(
            f"""
            SELECT
                content,
                embedding,
                vector::distance::knn() AS distance
            FROM
                {model.__name__}
            {query_where}
            ORDER BY
                distance ASC
            LIMIT
                $limit;
            """,  # nosec: B608
            **cast(Any, filter_variables),
            query=list(query_vector),
            limit=candidate_limit,
        )

        matching: list[Embedded[Model]] = []
        for record in rows:
            distance_value: SurrealValue = record["distance"]
            if not isinstance(distance_value, int | float):
                raise ValueError(f"Invalid SurrealDB vector distance: {distance_value!r}")

            if score_threshold is not None and (1.0 - float(distance_value)) < score_threshold:
                continue

            matching.append(
                Embedded(
                    value=model.from_mapping(cast(Mapping[str, Any], record["content"])),
                    vector=cast(Sequence[float], record["embedding"]),
                )
            )

        if not rerank:
            return tuple(element.value for element in matching[:result_limit])

        return tuple(
            matching[index].value
            for index in mmr_vector_similarity_search(
                query_vector=query_vector,
                values_vectors=[element.vector for element in matching],
                limit=result_limit,
            )
        )

    async def delete[Model: State](
        model: type[Model],
        /,
        *,
        requirements: AttributeRequirement[Model] | None = None,
        **extra: Any,
    ) -> None:
        _ = extra
        scoped_requirements: AttributeRequirement[Model] | None = _content_scoped_requirements(
            requirements
        )
        filter_clause, filter_variables = prepare_filter(scoped_requirements)
        try:
            await Surreal.execute(
                f"DELETE {model.__name__}{f' WHERE {filter_clause}' if filter_clause else ''};",
                **cast(Any, filter_variables),
            )
        except SurrealException as exc:
            # Deleting from a not-yet-created table should be a no-op.
            if "does not exist" not in str(exc).lower():
                raise

    return VectorIndex(
        indexing=index,
        searching=search,
        deleting=delete,
    )


def SurrealVectorIndex(
    *,
    mmr_multiplier: int = 8,
    search_effort: int = 40,
    vector_type: str = "F64",
    distance: str = "COSINE",
    efc: int | None = None,
    m: int | None = None,
) -> VectorIndex:
    """Backward-compatible alias for :func:`SurrealDBVectorIndex`."""

    return SurrealDBVectorIndex(
        mmr_multiplier=mmr_multiplier,
        search_effort=search_effort,
        vector_type=vector_type,
        distance=distance,
        efc=efc,
        m=m,
    )


def _surreal_option_token(
    value: str,
    /,
    *,
    name: str,
) -> str:
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        return value.upper()

    raise ValueError(f"Invalid SurrealDB {name}: {value!r}")


def _embedding_input(
    value: object,
    /,
) -> str | bytes:
    if isinstance(value, str):
        return value

    if isinstance(value, TextContent):
        return value.text

    if not isinstance(value, ResourceContent):
        raise TypeError("Selected attribute value has to be text or resource content")

    if value.mime_type.startswith("text"):
        return b64decode(value.data).decode()

    if not value.mime_type.startswith("image"):
        raise ValueError(f"{value.mime_type} embedding is not supported")

    return value.to_bytes()


def _content_scoped_requirements[Model: State](
    requirements: AttributeRequirement[Model] | None,
    /,
) -> AttributeRequirement[Model] | None:
    if requirements is None:
        return None

    return _content_scoped_requirement(requirements)


def _content_scoped_requirement[Model: State](
    requirement: AttributeRequirement[Model],
    /,
) -> AttributeRequirement[Model]:
    match requirement.operator:
        case "and":
            return _content_scoped_requirement(
                cast(AttributeRequirement[Model], requirement.lhs)
            ) & _content_scoped_requirement(cast(AttributeRequirement[Model], requirement.rhs))

        case "or":
            return _content_scoped_requirement(
                cast(AttributeRequirement[Model], requirement.lhs)
            ) | _content_scoped_requirement(cast(AttributeRequirement[Model], requirement.rhs))

        case _:
            scoped_lhs: str = _content_scoped_path(str(requirement.lhs))
            return cast(
                AttributeRequirement[Model],
                AttributeRequirement(
                    lhs=scoped_lhs,
                    operator=requirement.operator,
                    rhs=requirement.rhs,
                    check=lambda _value: None,
                ),
            )


def _content_scoped_path(
    path: str,
    /,
) -> str:
    if path.startswith("content."):
        return path

    return f"content.{path}"
