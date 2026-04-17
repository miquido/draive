import json
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, cast, final
from uuid import UUID

from haiway import Meta, MetaValues, Paginated, Pagination, cache, ctx
from haiway.postgres import Postgres, PostgresConnection, PostgresRow

from draive.multimodal.templates.repository import TemplatesRepository
from draive.multimodal.templates.types import TemplateDeclaration

__all__ = ("PostgresTemplatesRepository",)


@final
class PostgresTemplatesRepository:
    @staticmethod
    async def migrate() -> None:
        await PostgresConnection.execute(
            """
            CREATE TABLE IF NOT EXISTS templates (
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
            """
        )

    @staticmethod
    def prepare(
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
        """

        async def listing(
            pagination: Pagination | None,
            **extra: Any,
        ) -> Paginated[TemplateDeclaration]:
            _ = extra
            return await _list_template_declarations(pagination)

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

    __slots__ = ()

    def __init__(self) -> NoReturn:
        raise RuntimeError("PostgresTemplatesRepository instantiation is forbidden")


def _paginated_query_arguments(
    pagination: Pagination | None,
    /,
) -> tuple[Pagination, UUID | str | int | None, int]:
    pagination = pagination or Pagination.of(limit=32)
    if pagination.limit <= 0:
        return pagination, None, 0

    return (
        pagination,
        pagination.token,
        pagination.limit + 1,
    )


async def _list_template_declarations(
    pagination: Pagination | None,
    /,
) -> Paginated[TemplateDeclaration]:
    ctx.log_info("Listing templates...")
    pagination, token, fetch_limit = _paginated_query_arguments(pagination)
    if pagination.limit <= 0:
        return Paginated[TemplateDeclaration].of(
            (),
            pagination=pagination.with_token(None),
        )

    results: Sequence[PostgresRow]
    match token:
        case str() as identifier:
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
                    identifier < $1::TEXT

                ORDER BY
                    identifier DESC

                LIMIT $2::INTEGER;
                """,
                identifier,
                fetch_limit,
            )

        case int() as offset:
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
                    identifier DESC

                LIMIT $1::INTEGER
                OFFSET $2::INTEGER;
                """,
                fetch_limit,
                offset,
            )

        case None:
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
                    identifier DESC

                LIMIT $1::INTEGER;
                """,
                fetch_limit,
            )

        case _:
            raise ValueError("Invalid Postgres templates pagination token")

    page_results: Sequence[PostgresRow] = results[: pagination.limit]
    ctx.log_info(f"...{len(page_results)} results found!")
    next_token: str | None = None
    if len(results) > pagination.limit:
        next_token = f"{page_results[-1]['identifier']}"

    return Paginated[TemplateDeclaration].of(
        tuple(
            TemplateDeclaration(
                identifier=cast(str, result["identifier"]),
                description=cast(str | None, result["description"]),
                variables=json.loads(cast(str, result["variables"] or "{}")),
                meta=Meta.from_json(cast(str, result["meta"] or "{}")),
            )
            for result in page_results
        ),
        pagination=pagination.with_token(next_token),
    )
