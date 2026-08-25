import json
import re
import unicodedata
from base64 import b64decode
from collections.abc import Callable, Collection, Iterable, MutableSequence, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final, NoReturn, cast, final

from haiway import AttributePath, AttributeRequirement, State, ctx
from haiway.attributes import AttributesJSONEncoder
from haiway.postgres import Postgres, PostgresRow, PostgresValue

from draive.embedding import (
    Embedded,
    ImageEmbedding,
    TextEmbedding,
    VectorIndex,
    mmr_vector_similarity_search,
)
from draive.multimodal import TextContent
from draive.resources import ResourceContent
from draive.utils.attributes import attribute_path_segments

__all__ = (
    "PostgresVectorIndex",
    "postgres_identifier",
)

_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATH_SEGMENT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")
_WORD_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b\w+\b", re.UNICODE)


@final
class PostgresVectorIndex:
    """Factory namespace for Postgres-backed vector index implementations.

    Public API is provided through :meth:`prepare`, which builds and returns a
    :class:`VectorIndex` implementation using Postgres for persistence and search.

    Parameters
    ----------
    None
        This class is used as a static factory and does not accept initialization
        arguments.

    Returns
    -------
    PostgresVectorIndex
        A utility class representing the Postgres vector index factory API.
        Use :meth:`prepare` to obtain a runtime :class:`VectorIndex` instance:
        ``prepare(*, mmr_multiplier: int = 8) -> VectorIndex``.

    Raises
    ------
    AssertionError
        Raised by :meth:`prepare` when ``mmr_multiplier`` is less than ``1``.
    """

    @staticmethod
    def prepare(  # noqa: C901, PLR0915
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

        async def index[Model: State, Value: ResourceContent | TextContent | str](
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
                            INSERT INTO {postgres_identifier(model.__name__)} (
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
                            postgres_vector(embedded.vector),
                            embedded.value.to_json(),
                            embedded.meta.to_json(),
                            created_timestamp + timedelta(microseconds=idx),
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

                    FROM {postgres_identifier(model.__name__)}

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

            # bound in the pgvector text form - `query_vector` itself stays numeric
            # for the MMR reranking below
            arguments: Sequence[Sequence[PostgresValue] | PostgresValue] = (
                postgres_vector(query_vector),
            )
            # `<=>` is the cosine distance - matching both the `score_threshold`
            # conversion below and the recommended `vector_cosine_ops` index
            similarity_expression: str = f"embedding <=> ${len(arguments)}"

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
                    embedding::REAL[] AS embedding,
                    payload

                FROM {postgres_identifier(model.__name__)}

                {where_clause}
                ORDER BY {similarity_expression}
                LIMIT ${len(arguments)};
                """,  # nosec: B608
                *arguments,
            )

            if not rerank:
                return tuple(model.from_json(cast(str, result["payload"])) for result in results)

            # asyncpg has no codec for the pgvector `VECTOR` type - it is selected
            # cast to `REAL[]` (pgvector stores float4, so the cast is exact) which
            # decodes to the floats the reranking below computes on
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

        async def delete[Model: State](
            model: type[Model],
            /,
            *,
            requirements: AttributeRequirement[Model] | None = None,
            **extra: Any,
        ) -> None:
            if requirements is None:
                await Postgres.execute(
                    f"""
                    DELETE FROM {postgres_identifier(model.__name__)};
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
                DELETE FROM {postgres_identifier(model.__name__)}
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

    __slots__ = ()

    def __init__(self) -> NoReturn:
        raise RuntimeError("PostgresVectorIndex instantiation is forbidden")


def _resolve_requirement(  # noqa: C901, PLR0911
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
            accessor, resolved_arguments = _json_accessor(requirement.lhs, arguments)
            resolved_arguments = [*resolved_arguments, _json_value(requirement.rhs)]
            return (
                f"{accessor} = ${len(resolved_arguments)}::JSONB",
                resolved_arguments,
            )

        case "not_equal":
            accessor, resolved_arguments = _json_accessor(requirement.lhs, arguments)
            resolved_arguments = [*resolved_arguments, _json_value(requirement.rhs)]
            return (
                f"({accessor} IS DISTINCT FROM ${len(resolved_arguments)}::JSONB)",
                resolved_arguments,
            )

        case "contained_in":
            # haiway builds this operator with operands swapped - `lhs` holds the
            # collection of allowed values while `rhs` holds the attribute path
            accessor, resolved_arguments = _json_accessor(requirement.rhs, arguments)
            resolved_arguments = [
                *resolved_arguments,
                _json_values(cast(Iterable[Any], requirement.lhs)),
            ]
            return (
                f"{accessor} = ANY(${len(resolved_arguments)}::JSONB[])",
                resolved_arguments,
            )

        case "contains_any":
            accessor, resolved_arguments = _json_accessor(requirement.lhs, arguments)
            resolved_arguments = [
                *resolved_arguments,
                _json_values(cast(Iterable[Any], requirement.rhs)),
            ]
            return (
                # the interpolated accessor is fully parameterized
                f"EXISTS (SELECT 1 FROM jsonb_array_elements({accessor}) AS element"  # nosec: B608
                f" WHERE element = ANY(${len(resolved_arguments)}::JSONB[]))",
                resolved_arguments,
            )

        case "contains":
            accessor, resolved_arguments = _json_accessor(requirement.lhs, arguments)
            resolved_arguments = [*resolved_arguments, _json_value(requirement.rhs)]
            return (
                # the interpolated accessor is fully parameterized
                f"EXISTS (SELECT 1 FROM jsonb_array_elements({accessor}) AS element"  # nosec: B608
                f" WHERE element = ${len(resolved_arguments)}::JSONB)",
                resolved_arguments,
            )

        case "text_match":
            tokens: Sequence[str] = _text_tokens(str(requirement.rhs))
            if not tokens:
                return "TRUE", arguments  # nothing to match, same as haiway checks

            accessor, resolved_arguments = _json_text_accessor(requirement.lhs, arguments)
            clauses: MutableSequence[str] = []
            for token in tokens:
                resolved_arguments = [*resolved_arguments, token]
                # `\y` anchors at word boundaries, matching whole tokens
                # the same way the in memory requirement check does
                clauses.append(
                    f"{accessor} ~* ('\\y' || ${len(resolved_arguments)}::TEXT || '\\y')"
                )

            return f"({' AND '.join(clauses)})", resolved_arguments


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


def postgres_vector(
    vector: Sequence[float],
    /,
) -> str:
    """Render an embedding vector in the pgvector text representation.

    asyncpg has no codec for the pgvector `VECTOR` type, so a bound sequence is
    rejected with `expected str`. The text form is what the type accepts on input.
    """
    return f"[{','.join(repr(float(element)) for element in vector)}]"


def postgres_identifier(
    value: str,
    /,
) -> str:
    """Verify a table name before interpolating it into a statement.

    Postgres has no parameter form for table names, they have to be inlined,
    therefore each one has to be constrained to safe characters only.
    """
    if _IDENTIFIER_PATTERN.fullmatch(value):
        return value

    raise ValueError(f"Invalid Postgres identifier: {value!r}")


def _path_segments(
    path: Any,
    /,
) -> Sequence[str]:
    # the payload is stored serialized, attribute aliases have to be applied
    segments: Sequence[str] = attribute_path_segments(path)
    if not segments or any(
        _PATH_SEGMENT_PATTERN.fullmatch(segment) is None for segment in segments
    ):
        raise ValueError(f"Invalid Postgres attribute path: {path!r}")

    return segments


def _text_tokens(
    value: str,
    /,
) -> Sequence[str]:
    return tuple(_WORD_TOKEN_PATTERN.findall(unicodedata.normalize("NFC", value).casefold()))


def _json_accessor(
    path: Any,
    /,
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue],
) -> tuple[str, Sequence[Sequence[PostgresValue] | PostgresValue]]:
    # The path is bound as a parameter instead of being inlined within a literal,
    # additionally `#>` preserves the `jsonb` type - unlike the `#>>` text accessor
    # which forces each compared argument to be `text`, allowing only strings.
    resolved_arguments: Sequence[Sequence[PostgresValue] | PostgresValue] = [
        *arguments,
        _path_segments(path),
    ]
    return (f"payload #> ${len(resolved_arguments)}::TEXT[]", resolved_arguments)


def _json_text_accessor(
    path: Any,
    /,
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue],
) -> tuple[str, Sequence[Sequence[PostgresValue] | PostgresValue]]:
    # `#>>` unwraps the stored json string into text required by regex matching
    resolved_arguments: Sequence[Sequence[PostgresValue] | PostgresValue] = [
        *arguments,
        _path_segments(path),
    ]
    return (f"payload #>> ${len(resolved_arguments)}::TEXT[]", resolved_arguments)


def _json_value(
    value: Any,
    /,
) -> str:
    # `jsonb` arguments are bound as their JSON representation, encoded exactly
    # the way the compared payload was
    return json.dumps(value, cls=AttributesJSONEncoder)


def _json_values(
    values: Iterable[Any],
    /,
) -> Sequence[str]:
    return [_json_value(value) for value in values]
