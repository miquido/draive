from collections.abc import MutableSequence
from datetime import UTC, datetime, timedelta
from typing import Any, NoReturn, final
from uuid import UUID, uuid4

from haiway import Meta, MetaValues, ctx
from haiway.postgres import Postgres, PostgresConnection, PostgresRow

from draive.agents import AgentIdentity, AgentMemory, AgentThread
from draive.models import ModelContext, ModelContextElement, ModelInput, ModelOutput

__all__ = ("PostgresAgentMemory",)


@final
class PostgresAgentMemory:
    """PostgreSQL-backed agent memory.

    This utility exposes static helpers for schema migration and creating
    agent-scoped :class:`~draive.agents.state.AgentMemory` instances persisted
    in PostgreSQL. Recalled and remembered context is stored and read back
    as-is, keyed by the owning agent identity and the active conversation
    thread.

    Examples
    --------
    ```python
    from draive import ctx
    from draive.agents import AgentIdentity
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

            memory = PostgresAgentMemory.prepare(
                AgentIdentity.of(name="assistant"),
            )
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
                kind TEXT NOT NULL,
                payload JSONB NOT NULL,
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
                created ASC
            );
            """
        )

    @staticmethod
    def prepare(
        identity: AgentIdentity,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> AgentMemory:
        """Prepare agent-scoped memory operations backed by PostgreSQL.

        Parameters
        ----------
        identity : AgentIdentity
            Identity of the agent owning the persisted memory. Recalled and
            remembered context is isolated per agent URI.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the resulting memory instance.

        Returns
        -------
        AgentMemory
            A configured agent memory instance with recall and remember
            handlers bound to the provided agent identity.

        Raises
        ------
        Exception
            Raised by memory operations when PostgreSQL interactions fail.
        """
        agent_uri: str = identity.uri

        async def recall(
            thread: AgentThread,
            input: ModelInput,  # noqa: A002
            **extra: Any,
        ) -> ModelContext:
            return (
                *await _recall(
                    agent_uri=agent_uri,
                    thread_id=thread.identifier,
                ),
                input,
            )

        async def remember(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            await _remember(
                agent_uri=agent_uri,
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


def _element_from_row(
    row: PostgresRow,
    /,
) -> ModelContextElement:
    match row.get_str("kind", required=True):
        case "input":
            return ModelInput.from_json(row.get_str("payload", required=True))

        case "output":
            return ModelOutput.from_json(row.get_str("payload", required=True))

        case kind:
            raise ValueError(f"Unsupported agent memory payload kind: {kind}")


def _element_kind(
    element: ModelContextElement,
    /,
) -> str:
    if isinstance(element, ModelInput):
        return "input"

    else:
        return "output"


async def _recall(
    *,
    agent_uri: str,
    thread_id: UUID,
) -> ModelContext:
    return tuple(
        _element_from_row(row)
        for row in await Postgres.fetch(
            """
            SELECT
                kind,
                payload::TEXT

            FROM
                agent_memory

            WHERE
                agent_uri = $1::TEXT
            AND
                thread_id = $2::UUID

            ORDER BY
                created ASC;
            """,  # nosec: B608
            agent_uri,
            thread_id,
        )
    )


async def _remember(
    *,
    agent_uri: str,
    thread_id: UUID,
    context: ModelContext,
) -> None:
    async with Postgres.acquire_connection() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                SELECT
                    pg_advisory_xact_lock(
                        hashtext($1::TEXT),
                        hashtext(($2::UUID)::TEXT)
                    );
                """,  # nosec: B608
                agent_uri,
                thread_id,
            )

            await connection.execute(
                """
                DELETE FROM
                    agent_memory

                WHERE
                    agent_uri = $1::TEXT
                AND
                    thread_id = $2::UUID;
                """,  # nosec: B608
                agent_uri,
                thread_id,
            )

            # Fabricated, strictly increasing timestamps - `element.meta.created` is ignored
            # here because it isn't guaranteed to be set or monotonic across elements, and
            # `_recall` relies on `created` to reconstruct the original context order.
            timestamp: datetime = datetime.now(UTC)
            identifiers: MutableSequence[UUID] = []
            timestamps: MutableSequence[datetime] = []
            kinds: MutableSequence[str] = []
            payloads: MutableSequence[str] = []
            for position, element in enumerate(context):
                identifiers.append(uuid4())
                timestamps.append(timestamp + timedelta(microseconds=position))
                kinds.append(_element_kind(element))
                payloads.append(element.to_json())

            if identifiers:
                await connection.execute(
                    """
                    INSERT INTO
                        agent_memory (
                            agent_uri,
                            thread_id,
                            identifier,
                            created,
                            kind,
                            payload
                        )

                    SELECT
                        $1::TEXT,
                        $2::UUID,
                        value.identifier,
                        value.created,
                        value.kind,
                        value.payload::JSONB

                    FROM
                        UNNEST(
                            $3::UUID[],
                            $4::TIMESTAMPTZ[],
                            $5::TEXT[],
                            $6::TEXT[]
                        ) AS value(identifier, created, kind, payload);
                    """,  # nosec: B608
                    agent_uri,
                    thread_id,
                    identifiers,
                    timestamps,
                    kinds,
                    payloads,
                )
