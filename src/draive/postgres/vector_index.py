import re
from base64 import b64decode
from collections.abc import Callable, Collection, MutableMapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from haiway import AttributePath, AttributeRequirement, ctx
from haiway.postgres import Postgres, PostgresRow, PostgresValue

from draive.embedding import (
    Embedded,
    ImageEmbedding,
    TextEmbedding,
    mmr_vector_similarity_search,
)
from draive.multimodal import TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent
from draive.utils import VectorIndex

__all__ = ("PostgresVectorIndex",)


def PostgresVectorIndex(  # noqa: C901, PLR0915
    *,
    mmr_multiplier: int = 8,
) -> VectorIndex:
    """Create a Postgres-backed implementation of :class:`VectorIndex`.

    Parameters
    ----------
    mmr_multiplier
        Multiplier applied to ``limit`` to determine how many database rows are
        fetched before applying Maximal Marginal Relevance (MMR) re-ranking.

    Returns
    -------
    VectorIndex
        A VectorIndex implementation persisting entries in Postgres.

    Notes
    ------
    Example schema:
    ```
    CREATE TABLE IF NOT EXISTS your_table_name (
        id UUID NOT NULL DEFAULT gen_random_uuid(),
        embedding VECTOR(<dimension>) NOT NULL,
        payload JSONB NOT NULL,
        meta JSONB NOT NULL,
        created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    -- Recommended for cosine similarity
    CREATE INDEX IF NOT EXISTS your_table_name_embedding_idx
        ON your_table_name
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);

    ```
    """
    assert mmr_multiplier > 0  # nosec: B101

    table_names: MutableMapping[type[DataModel], str] = {}

    def resolve_table_name(model: type[DataModel]) -> str:
        nonlocal table_names
        if name := table_names.get(model):
            return name

        resolved: str = re.sub(r"(?<!^)(?=[A-Z])", "_", model.__name__).lower()
        table_names[model] = resolved
        return resolved

    async def index[Model: DataModel, Value: ResourceContent | TextContent | str](
        model: type[Model],
        /,
        *,
        attribute: Callable[[Model], Value] | AttributePath[Model, Value] | Value,
        values: Collection[Model],
        **extra: Any,
    ) -> None:
        assert isinstance(  # nosec: B101
            attribute, AttributePath | Callable
        ), f"Prepare parameter path by using {model.__name__}._.path.to.property"
        value_selector: Callable[[Model], Value] = cast(Callable[[Model], Value], attribute)

        selected_values: list[str | bytes] = []
        for value in values:
            selected: Value = value_selector(value)
            if isinstance(selected, str):
                selected_values.append(selected)

            elif isinstance(selected, TextContent):
                selected_values.append(selected.text)

            else:
                assert isinstance(selected, ResourceContent)  # nosec: B101
                if not selected.mime_type.startswith("image"):
                    raise ValueError(f"{selected.mime_type} embedding is not supported")

                selected_values.append(selected.to_bytes())

        embedded_values: Sequence[Embedded[Model]]
        if all(isinstance(value, str) for value in selected_values):
            embedded_values = [
                Embedded(
                    value=value,
                    vector=embedded.vector,
                    meta=embedded.meta,
                )
                for embedded, value in zip(
                    await TextEmbedding.embed_many(
                        selected_values,
                        **extra,
                    ),
                    values,
                    strict=True,
                )
            ]

        elif all(isinstance(value, bytes) for value in selected_values):
            embedded_values = [
                Embedded(
                    value=value,
                    vector=embedded.vector,
                    meta=embedded.meta,
                )
                for embedded, value in zip(
                    await ImageEmbedding.embed_many(
                        cast(list[bytes], selected_values),
                        **extra,
                    ),
                    values,
                    strict=True,
                )
            ]

        else:
            raise ValueError("Selected attribute values have to be the same type")

        created_timestamp: datetime = datetime.now(UTC)
        async with Postgres.acquire_connection() as connection:
            async with connection.transaction():
                for idx, embedded in enumerate(embedded_values):
                    await connection.execute(
                        f"""
                        INSERT INTO {resolve_table_name(model)} (
                            embedding,
                            payload,
                            meta,
                            created
                        )

                        VALUES (
                            $1::VECTOR,
                            $2::JSONB,
                            $3::JSONB,
                            $4::TIMESTAMPTZ
                        );
                        """,  # nosec: B608
                        embedded.vector,
                        embedded.value.to_json(),
                        embedded.meta.to_json(),
                        created_timestamp + timedelta(microseconds=idx),
                    )

        ctx.log_info("Vector index update completed.")

    async def search[Model: DataModel](  # noqa: C901, PLR0912
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
        assert query is not None or (query is None and score_threshold is None)  # nosec: B101
        where_clause: str
        arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
        if query is None:
            where_clause, arguments = resolve_requirements(requirements)
            if where_clause:
                where_clause = f"WHERE {where_clause}"

            parameters: Sequence[Sequence[PostgresValue] | PostgresValue] = [
                *arguments,
                limit or 8,
            ]
            results: Sequence[PostgresRow] = await Postgres.fetch(
                f"""
                SELECT
                    payload

                FROM {resolve_table_name(model)}

                {where_clause}
                ORDER BY created DESC
                LIMIT ${len(parameters)};
                """,  # nosec: B608
                *parameters,
            )

            return tuple(model.from_json(cast(str, result["payload"])) for result in results)

        query_vector: Sequence[float]
        if isinstance(query, str):
            embedded_query: Embedded[str] = await TextEmbedding.embed(query)
            query_vector = embedded_query.vector

        elif isinstance(query, TextContent):
            embedded_query: Embedded[str] = await TextEmbedding.embed(query.text)
            query_vector = embedded_query.vector

        elif isinstance(query, ResourceContent):
            if query.mime_type.startswith("image"):
                embedded_image: Embedded[bytes] = await ImageEmbedding.embed(query.to_bytes())
                query_vector = embedded_image.vector

            elif query.mime_type.startswith("text"):
                embedded_query: Embedded[str] = await TextEmbedding.embed(
                    b64decode(query.data).decode()
                )
                query_vector = embedded_query.vector

            else:
                raise ValueError(f"{query.mime_type} embedding is not supported")

        else:
            assert isinstance(query, Sequence)  # nosec: B101
            query_vector = query  # vector

        arguments: Sequence[Sequence[PostgresValue] | PostgresValue] = (query_vector,)
        similarity_expression: str = f"embedding <#> ${len(arguments)}"

        where_clause, arguments = resolve_requirements(requirements, arguments=arguments)

        if score_threshold is not None:
            arguments = (*arguments, 1.0 - float(score_threshold))
            threshold_clause: str = f"{similarity_expression} <= ${len(arguments)}"
            if where_clause:
                where_clause = f"WHERE {threshold_clause} AND ({where_clause})"

            else:
                where_clause = f"WHERE {threshold_clause}"

        elif where_clause:
            where_clause = f"WHERE {where_clause}"

        arguments = (*arguments, (limit or 8) * mmr_multiplier if rerank else (limit or 8))
        results: Sequence[PostgresRow] = await Postgres.fetch(
            f"""
            SELECT
                embedding,
                payload

            FROM {resolve_table_name(model)}

            {where_clause}
            ORDER BY {similarity_expression}
            LIMIT ${len(arguments)};
            """,  # nosec: B608
            *arguments,
        )

        if not rerank:
            return tuple(model.from_json(cast(str, result["payload"])) for result in results)

        matching: list[Embedded[Model]] = [
            Embedded[Model](
                vector=cast(Sequence[float], result["embedding"]),
                value=model.from_json(cast(str, result["payload"])),
            )
            for result in results
        ]
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
        if requirements is None:
            await Postgres.execute(
                f"""
                DELETE FROM {resolve_table_name(model)};
                """,  # nosec: B608
            )
            ctx.log_info(f"Removed all entries for {model.__name__}.")

            return

        where_clause: str
        arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
        where_clause, arguments = resolve_requirements(requirements)
        if where_clause:
            where_clause = f"WHERE {where_clause}"

        await Postgres.execute(
            f"""
            DELETE FROM {resolve_table_name(model)}
            {where_clause};
            """,  # nosec: B608
            *arguments,
        )
        ctx.log_info(f"Removed filtered entries for {model.__name__}.")

    return VectorIndex(
        indexing=index,
        searching=search,
        deleting=delete,
    )


def _resolve_requirement(  # noqa: PLR0911
    requirement: AttributeRequirement[Any],
    /,
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue],
) -> tuple[str, Sequence[Sequence[PostgresValue] | PostgresValue]]:
    resolved_arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    match requirement.operator:
        case "and":
            left_sql, partial_arguments = _resolve_requirement(
                requirement.lhs,
                arguments=arguments,
            )
            right_sql, resolved_arguments = _resolve_requirement(
                requirement.rhs,
                arguments=partial_arguments,
            )
            return f"({left_sql} AND {right_sql})", resolved_arguments

        case "or":
            left_sql, partial_arguments = _resolve_requirement(
                requirement.lhs,
                arguments=arguments,
            )
            right_sql, resolved_arguments = _resolve_requirement(
                requirement.rhs,
                arguments=partial_arguments,
            )
            return f"({left_sql} OR {right_sql})", resolved_arguments

        case "equal":
            resolved_arguments = [*arguments, requirement.rhs]
            return (
                f"{_scalar_accessor(str(requirement.lhs))} = ${len(resolved_arguments)}",
                resolved_arguments,
            )

        case "not_equal":
            resolved_arguments = [*arguments, requirement.rhs]
            return (
                f"({_scalar_accessor(str(requirement.lhs))}"
                f" IS DISTINCT FROM ${len(resolved_arguments)})",
                resolved_arguments,
            )

        case "contained_in":
            resolved_arguments = [*arguments, requirement.rhs]
            return (
                f"{_scalar_accessor(str(requirement.lhs))} = ANY(${len(resolved_arguments)})",
                resolved_arguments,
            )

        case "contains_any":
            resolved_arguments = [*arguments, requirement.rhs]
            return (
                "EXISTS (SELECT 1 FROM jsonb_array_elements_text("  # nosec: B608
                f"{_scalar_accessor(str(requirement.lhs))}) AS element"
                f" WHERE element = ANY(${len(resolved_arguments)}))",
                resolved_arguments,
            )

        case "contains":
            resolved_arguments = [*arguments, requirement.rhs]
            return (
                "EXISTS (SELECT 1 FROM jsonb_array_elements_text("  # nosec: B608
                f"{_scalar_accessor(str(requirement.lhs))} AS element"
                f" WHERE element = ${len(resolved_arguments)})",
                resolved_arguments,
            )

        case "text_match":
            # TODO: text match LIKE
            raise NotImplementedError("Not implemented yet")


def resolve_requirements(
    requirement: AttributeRequirement[Any] | None,
    /,
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue] = (),
) -> tuple[str, Sequence[Sequence[PostgresValue] | PostgresValue]]:
    if requirement is None:
        return ("", arguments)

    where_clause: str
    where_clause, arguments = _resolve_requirement(
        requirement,
        arguments=arguments,
    )

    return (where_clause, arguments)


def _path_literal(path: str) -> str:
    return f"'{{{','.join(path.lstrip('.').split('.'))}}}'"


def _scalar_accessor(path: str) -> str:
    return f"payload #>> {_path_literal(path)}"
