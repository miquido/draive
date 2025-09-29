from collections.abc import Sequence
from typing import Any, cast

from haiway import Meta, cache, ctx
from haiway.postgres import Postgres, PostgresRow

from draive.models.instructions import (
    InstructionsDeclaration,
    InstructionsRepository,
)
from draive.models.instructions.types import InstructionsArgumentDeclaration

__all__ = ("PostgresInstructionsRepository",)


# POSTGRES SCHEMA
#
# CREATE TABLE instructions (
#     name TEXT NOT NULL,
#     description TEXT DEFAULT NULL,
#     content TEXT NOT NULL,
#     arguments JSONB NOT NULL DEFAULT '[]'::jsonb,
#     meta JSONB NOT NULL DEFAULT '{}'::jsonb,
#     created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX instructions_name_created_idx
#     ON instructions (name, created DESC);
#


def PostgresInstructionsRepository(
    cache_limit: int = 32,
    cache_expiration: float = 600.0,  # 10 min
) -> InstructionsRepository:
    """Return a Postgres-backed instructions repository with caching.

    Parameters
    ----------
    cache_limit
        Maximum number of loaded instruction payloads cached concurrently.
    cache_expiration
        Lifetime in seconds for cached entries before reloading from Postgres.

    Returns
    -------
    InstructionsRepository
        Repository facade operating on the ``instructions`` Postgres table.
    """
    @cache(
        limit=1,
        expiration=cache_expiration,
    )
    async def listing(
        **extra: Any,
    ) -> Sequence[InstructionsDeclaration]:
        ctx.log_info("Listing instructions...")
        results: Sequence[PostgresRow] = await Postgres.fetch(
            """
            SELECT DISTINCT ON (name)
                name,
                description,
                arguments,
                meta

            FROM
                instructions

            ORDER BY
                name,
                created
            DESC;
            """
        )
        ctx.log_info(f"...{len(results)} results found!")

        return tuple(
            InstructionsDeclaration(
                name=cast(str, result["name"]),
                description=cast(str, result["description"] or ""),
                arguments=InstructionsArgumentDeclaration.from_json_array(
                    cast(str, result["arguments"] or "[]")
                ),
                meta=Meta.from_json(cast(str, result["meta"])),
            )
            for result in results
        )

    @cache(
        limit=cache_limit,
        expiration=cache_expiration,
    )
    async def load(
        name: str,
        /,
    ) -> str | None:
        ctx.log_info(f"Loading '{name}' instructions ...")
        result = await Postgres.fetch_one(
            """
            SELECT DISTINCT ON (name)
                content

            FROM
                instructions

            WHERE
                name = $1

            ORDER BY
                name,
                created
            DESC

            LIMIT 1;
            """,
            name,
        )

        if not result:
            ctx.log_info("...instructions not found!")
            return None

        ctx.log_info("...instructions loaded!")
        return cast(str, result["content"])

    async def loading(
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> str | None:
        return await load(name)

    async def defining(
        declaration: InstructionsDeclaration,
        content: str,
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Defining '{declaration.name}' instruction...")
        description = declaration.description or ""
        await Postgres.execute(
            """
            INSERT INTO
                instructions (
                    name,
                    description,
                    content,
                    arguments,
                    meta
                )

            VALUES
                (
                    $1,
                    $2,
                    $3,
                    $4::jsonb,
                    $5::jsonb
                );
            """,
            declaration.name,
            description,
            content,
            f"[{','.join(argument.to_json() for argument in declaration.arguments)}]",
            declaration.meta.to_json(),
        )
        ctx.log_info("...clearing cache...")
        await load.clear_cache()
        await listing.clear_cache()
        ctx.log_info("...instructions definition completed!")

    async def removing(
        name: str,
        meta: Meta,
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Removing '{name}' instructions...")
        await Postgres.execute(
            """
            DELETE FROM
                instructions

            WHERE
                name = $1;
            """,
            name,
        )

        ctx.log_info("...clearing cache...")
        await load.clear_cache()
        await listing.clear_cache()
        ctx.log_info("...instructions removal completed!")

    return InstructionsRepository(
        listing=listing,
        loading=loading,
        defining=defining,
        removing=removing,
    )
