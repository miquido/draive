import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from haiway import Meta, MetaValues, cache, ctx
from haiway.postgres import Postgres, PostgresRow

from draive.multimodal.templates.repository import TemplatesRepository
from draive.multimodal.templates.types import TemplateDeclaration

__all__ = ("PostgresTemplatesRepository",)


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

    @cache(
        limit=1,
        expiration=cache_expiration,
    )
    async def listing(
        **extra: Any,
    ) -> Sequence[TemplateDeclaration]:
        ctx.log_info("Listing templates...")

        results: Sequence[PostgresRow] = await Postgres.fetch(
            """
            SELECT DISTINCT ON (identifier)
                identifier::TEXT,
                description::TEXT,
                variables::JSONB,
                meta::JSONB,

            FROM
                templates

            ORDER BY
                identifier,
                created
            DESC;
            """
        )
        ctx.log_info(f"...{len(results)} results found!")

        return tuple(
            TemplateDeclaration(
                identifier=cast(str, result["identifier"]),
                description=cast(str | None, result["description"]),
                variables=json.loads(cast(str, result["variables"] or "{}")),
                meta=Meta.from_json(cast(str, result["meta"] or "{}")),
            )
            for result in results
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
                content::TEXT,

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
            description or None,
            content,
            json.dumps(variables),
            meta.to_json(),
        )
        ctx.log_info("...clearing cache...")
        await load.clear_cache()
        await listing.clear_cache()
        ctx.log_info("...template definition completed!")

    return TemplatesRepository(
        listing=listing,
        loading=loading,
        defining=defining,
        meta=Meta.of(meta if meta is not None else {"source": "postgres"}),
    )
