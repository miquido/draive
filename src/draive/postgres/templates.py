import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from haiway import Meta, MetaValues, Paginated, Pagination, cache, ctx
from haiway.postgres import Postgres, PostgresRow

from draive.multimodal.templates.repository import TemplatesRepository
from draive.multimodal.templates.types import TemplateDeclaration

__all__ = ("PostgresTemplatesRepository",)


def _declaration_from_row(
    row: PostgresRow,
    /,
) -> TemplateDeclaration:
    return TemplateDeclaration(
        identifier=cast(str, row["identifier"]),
        description=cast(str | None, row["description"]),
        variables=json.loads(cast(str, row["variables"] or "{}")),
        meta=Meta.from_json(cast(str, row["meta"] or "{}")),
    )


def _templates_pagination_token(
    pagination: Pagination,
    /,
) -> str | int | None:
    if pagination.token is None:
        return None

    if isinstance(pagination.token, str):
        if not pagination.token.startswith("templates:"):
            raise ValueError("Invalid postgres templates pagination token")

        token: str = pagination.token.split(":", 1)[1]
        if token.startswith("cursor:"):
            cursor: str = token.removeprefix("cursor:")
            if cursor:
                return cursor

            raise ValueError("Invalid postgres templates pagination token")

        try:
            return max(int(token), 0)

        except ValueError as exc:
            raise ValueError("Invalid postgres templates pagination token") from exc

    if isinstance(pagination.token, int):
        return max(pagination.token, 0)

    raise ValueError("Invalid postgres templates pagination token")


def _paginated_query_arguments(
    pagination: Pagination | None,
    /,
) -> tuple[Pagination, str | int | None, int]:
    resolved_pagination: Pagination = pagination or Pagination.of(limit=32)
    if resolved_pagination.limit <= 0:
        return resolved_pagination, None, 0

    return (
        resolved_pagination,
        _templates_pagination_token(resolved_pagination),
        resolved_pagination.limit + 1,
    )


def PostgresTemplatesRepository(
    cache_limit: int = 32,
    cache_expiration: float = 600.0,  # 10 min
    meta: Meta | MetaValues | None = None,
) -> TemplatesRepository:
    """Return a Postgres-backed templates repository with caching.

    Parameters
    ----------
    cache_limit
        Maximum number of loaded template payloads cached concurrently.
    cache_expiration
        Lifetime in seconds for cached entries before reloading from Postgres.

    Returns
    -------
    TemplatesRepository
        Repository facade operating on the ``templates`` Postgres table.

    Notes
    ------
    Example schema:
    ```
    CREATE TABLE templates (
        identifier TEXT NOT NULL,
        description TEXT DEFAULT NULL,
        content TEXT NOT NULL,
        variables JSONB NOT NULL DEFAULT '{}'::jsonb,
        meta JSONB NOT NULL DEFAULT '{}'::jsonb,
        created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (identifier, created)
    );

    CREATE INDEX IF NOT EXISTS
        templates_idx

    ON
        templates (identifier, created DESC);
    ```
    """

    async def listing(
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[TemplateDeclaration]:
        _ = extra
        ctx.log_info("Listing templates...")
        resolved_pagination, pagination_token, fetch_limit = _paginated_query_arguments(pagination)

        if resolved_pagination.limit <= 0:
            return Paginated[TemplateDeclaration].of(
                (),
                pagination=resolved_pagination.with_token(None),
            )

        results: Sequence[PostgresRow]
        if isinstance(pagination_token, str):
            results = await Postgres.fetch(
                """
                WITH latest_templates AS (
                    SELECT DISTINCT ON (identifier)
                        identifier::TEXT,
                        description::TEXT,
                        variables::JSONB,
                        meta::JSONB

                    FROM
                        templates

                    ORDER BY
                        identifier,
                        created
                    DESC
                )

                SELECT
                    identifier::TEXT,
                    description::TEXT,
                    variables::JSONB,
                    meta::JSONB

                FROM
                    latest_templates

                WHERE
                    identifier > $1::TEXT

                ORDER BY
                    identifier

                LIMIT $2::INTEGER;
                """,
                pagination_token,
                fetch_limit,
            )

        elif isinstance(pagination_token, int):
            results = await Postgres.fetch(
                """
                WITH latest_templates AS (
                    SELECT DISTINCT ON (identifier)
                        identifier::TEXT,
                        description::TEXT,
                        variables::JSONB,
                        meta::JSONB

                    FROM
                        templates

                    ORDER BY
                        identifier,
                        created
                    DESC
                )

                SELECT
                    identifier::TEXT,
                    description::TEXT,
                    variables::JSONB,
                    meta::JSONB

                FROM
                    latest_templates

                ORDER BY
                    identifier

                LIMIT $1::INTEGER
                OFFSET $2::INTEGER;
                """,
                fetch_limit,
                pagination_token,
            )

        else:
            results = await Postgres.fetch(
                """
                WITH latest_templates AS (
                    SELECT DISTINCT ON (identifier)
                        identifier::TEXT,
                        description::TEXT,
                        variables::JSONB,
                        meta::JSONB

                    FROM
                        templates

                    ORDER BY
                        identifier,
                        created
                    DESC
                )

                SELECT
                    identifier::TEXT,
                    description::TEXT,
                    variables::JSONB,
                    meta::JSONB

                FROM
                    latest_templates

                ORDER BY
                    identifier

                LIMIT $1::INTEGER;
                """,
                fetch_limit,
            )
        ctx.log_info(f"...{len(results)} results found!")

        declarations: Sequence[TemplateDeclaration] = tuple(
            _declaration_from_row(result) for result in results[: resolved_pagination.limit]
        )
        next_token: str | None = None
        if len(results) > resolved_pagination.limit and declarations:
            next_token = f"templates:cursor:{declarations[-1].identifier}"

        return Paginated[TemplateDeclaration].of(
            declarations,
            pagination=resolved_pagination.with_token(next_token),
        )

    @cache(
        limit=cache_limit,
        expiration=cache_expiration,
    )
    async def load(
        identifier: str,
        /,
    ) -> str | None:
        ctx.log_info(f"Loading '{identifier}' template ...")
        result = await Postgres.fetch_one(
            """
            SELECT DISTINCT ON (identifier)
                content::TEXT

            FROM
                templates

            WHERE
                identifier = $1::TEXT

            ORDER BY
                identifier,
                created
            DESC

            LIMIT 1;
            """,
            identifier,
        )

        if not result:
            ctx.log_info("...template not found!")
            return None

        ctx.log_info("...template loaded!")
        return cast(str, result["content"])

    async def loading(
        identifier: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        return await load(identifier)

    async def defining(
        identifier: str,
        description: str | None,
        content: str,
        variables: Mapping[str, str],
        meta: Meta,
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Defining '{identifier}' template...")
        await Postgres.execute(
            """
            INSERT INTO
                templates (
                    identifier,
                    description,
                    content,
                    variables,
                    meta
                )

            VALUES
                (
                    $1::TEXT,
                    $2::TEXT,
                    $3::TEXT,
                    $4::JSONB,
                    $5::JSONB
                );
            """,
            identifier,
            description,
            content,
            json.dumps(variables),
            meta.to_json(),
        )
        ctx.log_info("...clearing cache...")
        await load.clear_cache()
        ctx.log_info("...template definition completed!")

    return TemplatesRepository(
        listing=listing,
        loading=loading,
        defining=defining,
        meta=Meta.of(meta if meta is not None else {"source": "postgres"}),
    )
