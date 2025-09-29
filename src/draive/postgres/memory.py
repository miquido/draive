import json
from collections.abc import Generator, Mapping, Sequence
from typing import Any, cast

from haiway import BasicValue, Map, ctx
from haiway.postgres import Postgres, PostgresConnection, PostgresRow, PostgresValue

from draive.models import ModelMemory, ModelMemoryRecall
from draive.models.types import ModelContextElement, ModelInput, ModelOutput

__all__ = ("PostgresModelMemory",)


# POSTGRES SCHEMA
#
# CREATE TABLE memories (
#     identifier TEXT NOT NULL,
#     variables JSONB NOT NULL DEFAULT '{}'::jsonb,
#     created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX memories_identifier_created_idx
#     ON memories (identifier, created DESC);
#
# CREATE TABLE memories_elements (
#     identifier TEXT NOT NULL REFERENCES memories (identifier) ON DELETE CASCADE,
#     content JSONB NOT NULL,
#     created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX memories_elements_identifier_idx
#     ON memories_elements (identifier, created DESC);
#


def PostgresModelMemory(
    identifier: str,
    *,
    recall_limit: int | None = None,
) -> ModelMemory:
    """Create a model memory bound to a Postgres-backed storage.

    Parameters
    ----------
    identifier
        Key identifying the memory records grouping in the ``memories`` tables.
    recall_limit
        Optional maximum number of context elements returned during recall.

    Returns
    -------
    ModelMemory
        Memory interface persisting variables and context elements in Postgres.

    Raises
    ------
    AssertionError
        If ``recall_limit`` is provided with a non-positive value.
    """
    assert recall_limit is None or recall_limit > 0  # nosec: B101

    async def recall(
        **extra: Any,
    ) -> ModelMemoryRecall:
        ctx.log_info(f"Recalling memory for {identifier}...")

        variables: Mapping[str, BasicValue] = await _load_variables(
            identifier=identifier,
        )

        ctx.log_info("...loading context...")

        context_elements: Sequence[ModelContextElement] = await _load_context(
            identifier=identifier,
            limit=extra.get("limit", recall_limit),
        )

        ctx.log_info(f"...{len(context_elements)} context elements recalled!")

        return ModelMemoryRecall(
            context=context_elements,
            variables=variables,
        )

    async def remember(
        *items: ModelContextElement,
        variables: Mapping[str, BasicValue] | None = None,
        **extra: Any,
    ) -> None:
        if not items and variables is None:
            return ctx.log_info(f"No content to remember for {identifier}, skipping!")

        ctx.log_info(f"Remembering content for {identifier}...")

        async with Postgres.acquire_connection():
            async with PostgresConnection.transaction():
                if variables is not None:
                    ctx.log_info(f"...remembering {len(variables)} variables...")
                    await PostgresConnection.execute(
                        """
                        INSERT INTO
                            memories (
                                identifier,
                                variables
                            )

                        VALUES (
                            $1,
                            $2::jsonb
                        );
                        """,
                        identifier,
                        json.dumps(variables),
                    )

                ctx.log_info(f"...remembering {len(items)} context elements...")
                for element in items:
                    await PostgresConnection.execute(
                        """
                        INSERT INTO
                            memories_elements (
                                identifier,
                                content
                            )

                        VALUES (
                            $1,
                            $2::jsonb
                        );
                        """,
                        identifier,
                        element.to_json(),
                    )

        ctx.log_info("...memory persisted!")

    async def maintenance(
        variables: Mapping[str, BasicValue] | None = None,
        **extra: Any,
    ) -> None:
        ctx.log_info(f"Performing maintenance for {identifier}...")

        async with Postgres.acquire_connection():
            async with PostgresConnection.transaction():
                if variables is not None:
                    ctx.log_info(f"...remembering {len(variables)} variables...")
                    await PostgresConnection.execute(
                        """
                            INSERT INTO
                                memories (
                                    identifier,
                                    variables
                                )

                            VALUES (
                                $1,
                                $2::jsonb
                            );
                            """,
                        identifier,
                        json.dumps(variables),
                    )

                else:
                    ctx.log_info("...ensuring memory entry exists...")
                    existing: PostgresRow | None = await PostgresConnection.fetch_one(
                        """
                            SELECT
                                1

                            FROM
                                memories

                            WHERE
                                identifier = $1

                            LIMIT 1;
                            """,
                        identifier,
                    )

                    if existing is None:
                        ctx.log_info("... creating new memories...")
                        await PostgresConnection.execute(
                            """
                                INSERT INTO
                                    memories (
                                        identifier,
                                        variables
                                    )

                                VALUES (
                                    $1,
                                    $2::jsonb
                                );
                                """,
                            identifier,
                            json.dumps({}),
                        )

        ctx.log_info("...maintenance completed!")

    return ModelMemory(
        recall=recall,
        remember=remember,
        maintenance=maintenance,
    )


async def _load_context(
    *,
    identifier: str,
    limit: int | None,
) -> Sequence[ModelContextElement]:
    statement: str
    parameters: Sequence[PostgresValue]
    if limit:
        statement = """
        SELECT
            content

        FROM
            memories_elements

        WHERE
            identifier = $1

        ORDER BY
            created
        DESC

        LIMIT $2;
        """
        parameters = (identifier, limit)

    else:
        statement = """
        SELECT
            content

        FROM
            memories_elements

        WHERE
            identifier = $1

        ORDER BY
            created
        DESC;
        """
        parameters = (identifier,)

    return tuple(
        reversed(
            tuple(
                _decode_context(
                    await Postgres.fetch(
                        statement,
                        *parameters,
                    )
                )
            )
        )
    )


def _decode_context(
    rows: Sequence[PostgresRow],
    /,
) -> Generator[ModelContextElement]:
    for row in rows:
        decoded: Mapping[str, BasicValue] = json.loads(cast(str, row["content"]))
        match decoded:
            case {"type": "model_input"}:
                yield ModelInput.from_mapping(decoded)

            case {"type": "model_output"}:
                yield ModelOutput.from_mapping(decoded)

            case other:
                raise ValueError(f"Invalid model context element: {other}")


async def _load_variables(
    *,
    identifier: str,
) -> Mapping[str, BasicValue]:
    row: PostgresRow | None = await Postgres.fetch_one(
        """
        SELECT DISTINCT ON (identifier)
            variables

        FROM
            memories

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

    if row is None:
        return Map()

    return cast(Mapping[str, BasicValue], json.loads(cast(str, row["variables"])))
