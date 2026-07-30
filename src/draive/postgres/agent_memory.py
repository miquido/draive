import json
from typing import Any, NoReturn, final
from uuid import UUID, uuid4

from haiway import Meta, MetaValues, ctx
from haiway.postgres import Postgres, PostgresConnection, PostgresRow

from draive.agents import AgentMemory, AgentThread
from draive.models import ModelContext, ModelContextElement, ModelInput, ModelOutput

__all__ = ("PostgresAgentMemory",)


@final
class PostgresAgentMemory:
    """PostgreSQL-backed agent memory.

    This utility exposes static helpers for schema migration and creating
    :class:`~draive.agents.state.AgentMemory` instances persisted in
    PostgreSQL, keyed by the executing agent URI and thread identifier
    carried on the ``AgentThread`` passed to each memory operation. A single
    instance can serve multiple agents.

    Context is stored as immutable snapshots: every remember inserts the full
    context as a new snapshot row and recall reads back the latest one whole.
    Agents may transform their context arbitrarily between recall and
    remember (compaction, summarization, replacement) - whatever is
    remembered becomes the next recalled context, exactly as provided.
    Previous snapshots are never modified or deleted and remain available
    for tracking and verification. Persistence is lock-free and write-only:
    concurrent remembers within the same thread each store their own
    snapshot and recall picks the latest one by creation timestamp -
    recorded at statement execution time with microsecond resolution, with
    the snapshot identifier as a stable tie-break. Snapshot history grows
    without bound over the lifetime of a thread.

    Examples
    --------
    ```python
    from draive import ctx
    from haiway.postgres import Postgres
    from draive.postgres.agent_memory import PostgresAgentMemory

    async def bootstrap_memory() -> None:
        async with ctx.scope("agent-memory"):
            # `migrate()` calls `PostgresConnection.execute`, which reads the
            # active connection from context - it must run with a connection
            # acquired and bound via `ctx.updating`, not just a pool in scope.
            async with Postgres.acquire_connection() as connection:
                with ctx.updating(connection):
                    await PostgresAgentMemory.migrate()

            memory = PostgresAgentMemory.instance()
    ```
    """

    @staticmethod
    async def migrate() -> None:
        """Create database structures required by agent memory.

        This asynchronous method creates the `agent_memory` table and its
        supporting index when they do not already exist.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Completes when the schema migration statements finish.

        Raises
        ------
        Exception
            Raised when PostgreSQL command execution fails, for example due to
            connection or database-level errors.
        """
        await PostgresConnection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_memory (
                agent_uri TEXT NOT NULL,
                thread_id UUID NOT NULL,
                identifier UUID NOT NULL,
                context JSONB NOT NULL,
                created TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (agent_uri, thread_id, identifier)
            );
            """
        )
        await PostgresConnection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                agent_memory_idx

            ON agent_memory (
                agent_uri,
                thread_id,
                created DESC,
                identifier DESC
            );
            """
        )

    @staticmethod
    def instance(
        *,
        meta: Meta | MetaValues | None = None,
    ) -> AgentMemory:
        """Create memory operations backed by PostgreSQL.

        The executing agent is resolved per operation from the provided
        ``AgentThread``, so a single memory instance can serve multiple
        agents - recalled and remembered context is isolated per agent URI
        and thread.

        Parameters
        ----------
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the resulting memory instance.

        Returns
        -------
        AgentMemory
            A configured agent memory instance with recall and remember
            handlers scoped by the executing agent context.

        Raises
        ------
        Exception
            Raised by memory operations when PostgreSQL interactions fail.
        """

        async def recall(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> ModelContext:
            return (
                *await _recall(
                    agent_uri=thread.agent_uri,
                    thread_id=thread.identifier,
                ),
                *context,
            )

        async def remember(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            await _remember(
                agent_uri=thread.agent_uri,
                thread_id=thread.identifier,
                context=context,
            )

            ctx.log_debug("...agent memory persisted.")

        return AgentMemory(
            recalling=recall,
            remembering=remember,
            meta=Meta.of(meta),
        )

    __slots__ = ()

    def __init__(self) -> NoReturn:
        raise RuntimeError("PostgresAgentMemory instantiation is forbidden")


def _element_from_value(
    value: Any,
    /,
) -> ModelContextElement:
    # elements serialize their own `kind` discriminator - dispatch on it directly
    match value:
        case {"kind": "input"}:
            return ModelInput.from_mapping(value)

        case {"kind": "output"}:
            return ModelOutput.from_mapping(value)

        case other:
            raise ValueError(f"Unsupported agent memory snapshot element: {other}")


async def _recall(
    *,
    agent_uri: str,
    thread_id: UUID,
) -> ModelContext:
    snapshot: PostgresRow | None = await Postgres.fetch_one(
        """
        SELECT
            context::TEXT

        FROM
            agent_memory

        WHERE
            agent_uri = $1::TEXT
        AND
            thread_id = $2::UUID

        ORDER BY
            created DESC,
            identifier DESC

        LIMIT 1;
        """,  # nosec: B608
        agent_uri,
        thread_id,
    )
    if snapshot is None:
        return ()

    return tuple(
        _element_from_value(value)
        for value in json.loads(snapshot.get_str("context", required=True))
    )


async def _remember(
    *,
    agent_uri: str,
    thread_id: UUID,
    context: ModelContext,
) -> None:
    # write-only: a new snapshot is inserted as-is, previous snapshots stay
    # untouched for tracking and verification; recall picks the latest snapshot
    # by creation timestamp, with the identifier as a stable tie-break
    await Postgres.execute(
        """
        INSERT INTO
            agent_memory (
                agent_uri,
                thread_id,
                identifier,
                context,
                created
            )

        VALUES (
            $1::TEXT,
            $2::UUID,
            $3::UUID,
            $4::JSONB,
            clock_timestamp()
        );
        """,  # nosec: B608
        agent_uri,
        thread_id,
        uuid4(),
        # elements carry their own `kind` discriminator - store their JSON directly
        f"[{','.join(element.to_json() for element in context)}]",
    )
