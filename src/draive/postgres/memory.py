import json
from collections.abc import Generator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

from haiway import BasicValue, Map, Meta, MetaValues, ctx
from haiway.postgres import Postgres, PostgresRow

from draive.models import (
    ModelContextElement,
    ModelInput,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
)

__all__ = ("PostgresModelMemory",)


def PostgresModelMemory(
    identifier: UUID,
    *,
    recall_limit: int = 0,
    meta: Meta | MetaValues | None = None,
) -> ModelMemory:
    """Create a model memory bound to a Postgres-backed storage.

    Parameters
    ----------
    identifier: UUID
        Key identifying the memory records grouping in the ``memories`` tables.
    recall_limit: int
        Optional maximum number of context elements returned during recall.

    Returns
    -------
    ModelMemory
        Memory interface persisting variables and context elements in Postgres.

    Notes
    ------
    Example schema:
    ```
    CREATE TABLE memories (
        identifier UUID NOT NULL,
        created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (identifier)
    );

    CREATE TABLE memories_variables (
        memories UUID NOT NULL REFERENCES memories (identifier) ON DELETE CASCADE,
        variables JSONB NOT NULL DEFAULT '{}'::jsonb,
        created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS
        memories_variables_idx

    ON
        memories_variables (memories, created DESC);

    CREATE TABLE memories_elements (
        identifier UUID NOT NULL DEFAULT gen_random_uuid(),
        memories UUID NOT NULL REFERENCES memories (identifier) ON DELETE CASCADE,
        content JSONB NOT NULL,
        created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (identifier)
    );

    CREATE INDEX IF NOT EXISTS
        memories_elements_idx

    ON
        memories_elements (
            memories,
            created DESC
        );
    ```
    """
    assert recall_limit >= 0  # nosec: B101

    async def recall(
        **extra: Any,
    ) -> ModelMemoryRecall:
        ctx.log_info(f"Recalling memory for {identifier}...")

        ctx.log_info("...loading variables...")
        variables: Mapping[str, BasicValue] = await _load_variables(
            identifier=identifier,
        )

        ctx.log_info("...loading context...")

        context_elements: Sequence[ModelContextElement] = await _load_context(
            identifier=identifier,
            limit=recall_limit,
        )

        ctx.log_info(f"...{len(context_elements)} context elements recalled!")
        ctx.record_info(
            event="postgres.memory.recall",
            attributes={
                "identifier": str(identifier),
                "limit": recall_limit,
                "elements": len(context_elements),
                "variables": len(variables),
            },
        )

        return ModelMemoryRecall(
            context=context_elements,
            variables=variables,
        )

    async def remember(
        *elements: ModelContextElement,
        variables: Mapping[str, BasicValue] | None = None,
        **extra: Any,
    ) -> None:
        if not elements and variables is None:
            return ctx.log_info(f"No content to remember for {identifier}, skipping!")

        ctx.log_info(f"Remembering content for {identifier}...")

        async with Postgres.acquire_connection() as connection:
            async with connection.transaction():
                if variables is not None:
                    ctx.log_info(f"...remembering {len(variables)} variables...")
                    await connection.execute(
                        """
                        INSERT INTO
                            memories_variables (
                                memories,
                                variables
                            )

                        VALUES (
                            $1::UUID,
                            $2::JSONB
                        );
                        """,
                        identifier,
                        json.dumps(variables),
                    )

                ctx.log_info(f"...remembering {len(elements)} context elements...")
                timestamp: datetime = datetime.now(UTC)
                for idx, element in enumerate(elements):
                    created: datetime = timestamp + timedelta(microseconds=idx)

                    await connection.execute(
                        """
                        INSERT INTO
                            memories_elements (
                                identifier,
                                memories,
                                content,
                                created
                            )

                        VALUES (
                            $1::UUID,
                            $2::UUID,
                            $3::JSONB,
                            $4::TIMESTAMPTZ
                        )

                        ON CONFLICT (identifier) DO UPDATE
                            SET
                                content = EXCLUDED.content

                            WHERE
                                memories_elements.memories = EXCLUDED.memories;
                        """,
                        element.identifier,
                        identifier,
                        element.to_json(),
                        created,
                    )

        ctx.record_info(
            event="postgres.memory.remember",
            attributes={
                "identifier": str(identifier),
                "elements": len(elements),
                "variables": len(variables) if variables is not None else 0,
            },
        )
        ctx.log_info("...memory persisted!")

    async def maintenance(
        variables: Mapping[str, BasicValue] | None = None,
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Performing maintenance for {identifier}...")
        async with Postgres.acquire_connection() as connection:
            async with connection.transaction():
                ctx.log_info("...ensuring memory entry exists...")
                await connection.execute(
                    """
                        INSERT INTO
                            memories (
                                identifier
                            )

                        VALUES (
                            $1::UUID
                        )

                        ON CONFLICT (identifier)
                        DO NOTHING;
                        """,
                    identifier,
                )

                if variables is not None:
                    ctx.log_info(f"...remembering {len(variables)} variables...")
                    await connection.execute(
                        """
                        INSERT INTO
                            memories_variables (
                                memories,
                                variables
                            )

                        VALUES (
                            $1::UUID,
                            $2::JSONB
                        );
                        """,
                        identifier,
                        json.dumps(variables),
                    )

        ctx.log_info("...maintenance completed!")

    return ModelMemory(
        recalling=recall,
        remembering=remember,
        maintaining=maintenance,
        meta=Meta.of(meta if meta is not None else {"source": "postgres"}),
    )


async def _load_context(
    *,
    identifier: UUID,
    limit: int,
) -> Sequence[ModelContextElement]:
    rows: Sequence[PostgresRow]
    if limit:
        rows = await Postgres.fetch(
            """
            SELECT
                content::JSONB

            FROM
                memories_elements

            WHERE
                memories = $1::UUID

            ORDER BY
                created DESC

            LIMIT $2;
            """,
            identifier,
            limit,
        )

    else:
        rows = await Postgres.fetch(
            """
            SELECT
                content::JSONB

            FROM
                memories_elements

            WHERE
                memories = $1::UUID

            ORDER BY
                created DESC;
            """,
            identifier,
        )

    return tuple(_decode_context(rows))


def _decode_context(
    rows: Sequence[PostgresRow],
    /,
) -> Generator[ModelContextElement]:
    trim_prefix: bool = True
    skip_output: bool = True
    for row in reversed(rows):
        decoded: Mapping[str, BasicValue] = json.loads(cast(str, row["content"]))
        match decoded.get("type"):
            case "model_input":
                model_input: ModelInput = ModelInput.from_mapping(decoded)
                if trim_prefix:
                    if model_input.contains_tools:
                        skip_output = True
                        continue

                    else:
                        skip_output = False
                        trim_prefix = False

                yield model_input

            case "model_output":
                model_output: ModelOutput = ModelOutput.from_mapping(decoded)
                if trim_prefix:
                    if model_output.contains_tools:
                        continue

                    elif skip_output:
                        continue

                    else:
                        trim_prefix = False

                yield model_output

            case other:
                raise ValueError(f"Invalid model context element: {other}")


async def _load_variables(
    *,
    identifier: UUID,
) -> Mapping[str, BasicValue]:
    row: PostgresRow | None = await Postgres.fetch_one(
        """
        SELECT
            variables::JSONB

        FROM
            memories_variables

        WHERE
            memories = $1::UUID

        ORDER BY created DESC

        LIMIT 1;
        """,
        identifier,
    )

    if row is None:
        return Map()

    return cast(Mapping[str, BasicValue], json.loads(cast(str, row["variables"])))
