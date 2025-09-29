import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from haiway import BasicValue, ConfigurationRepository, cache, ctx
from haiway.postgres import Postgres, PostgresRow

__all__ = ("PostgresConfigurationRepository",)

# POSTGRES SCHEMA
#
# CREATE TABLE configurations (
#     identifier TEXT NOT NULL,
#     content JSONB NOT NULL,
#     created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX configurations_identifier_created_idx
#     ON configurations (identifier, created DESC);
#


def PostgresConfigurationRepository(
    cache_limit: int = 32,
    cache_expiration: float = 600.0,  # 10 min
) -> ConfigurationRepository:
    """Return a repository storing configuration snapshots in Postgres.

    Parameters
    ----------
    cache_limit
        Maximum number of configuration documents kept in the in-memory cache.
    cache_expiration
        Lifetime in seconds for cached entries before a fresh query is issued.

    Returns
    -------
    ConfigurationRepository
        Repository facade backed by the ``configurations`` Postgres table.
    """
    @cache(
        limit=1,
        expiration=cache_expiration,
    )
    async def listing(
        **extra: Any,
    ) -> Sequence[str]:
        ctx.log_info("Listing configurations...")
        results: Sequence[PostgresRow] = await Postgres.fetch(
            """
                SELECT DISTINCT ON (identifier)
                    identifier

                FROM
                    configurations

                ORDER BY
                    identifier,
                    created
                DESC;
                """
        )
        ctx.log_info(f"...{len(results)} results found!")
        return tuple(cast(str, record["identifier"]) for record in results)

    @cache(
        limit=cache_limit,
        expiration=cache_expiration,
    )
    async def loading(
        identifier: str,
        **extra: Any,
    ) -> Mapping[str, BasicValue] | None:
        ctx.log_info(f"Loading configuration for {identifier}...")
        loaded: PostgresRow | None = await Postgres.fetch_one(
            """
            SELECT DISTINCT ON (identifier)
                identifier,
                content

            FROM
                configurations

            WHERE
                identifier = $1

            ORDER BY
                identifier,
                created
            DESC

            LIMIT 1;
            """,
            identifier,
        )

        if loaded is None:
            ctx.log_info("...configuration not found!")
            return None

        ctx.log_info("...decoding configuration...")
        content: Mapping[str, BasicValue] = json.loads(cast(str, loaded["content"]))
        ctx.log_info("...configuration loaded!")
        return content

    async def define(
        identifier: str,
        value: Mapping[str, BasicValue],
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Defining configuration {identifier}...")
        await Postgres.execute(
            """
            INSERT INTO
                configurations (
                    identifier,
                    content
                )

            VALUES (
                $1,
                $2::jsonb
            );
            """,
            identifier,
            json.dumps(value),
        )
        ctx.log_info("...clearing cache...")
        await loading.clear_cache()
        await listing.clear_cache()
        ctx.log_info("...configuration definition completed!")

    async def removing(
        identifier: str,
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Removing configuration {identifier}...")
        await Postgres.execute(
            """
            DELETE FROM
                configurations

            WHERE
                identifier = $1;
            """,
            identifier,
        )
        ctx.log_info("...clearing cache...")
        await loading.clear_cache()
        await listing.clear_cache()
        ctx.log_info("...configuration removal completed!")

    return ConfigurationRepository(
        listing=listing,
        loading=loading,
        defining=define,
        removing=removing,
    )
