import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, NoReturn, cast, final
from uuid import UUID

from haiway import Paginated, Pagination, ctx
from haiway.postgres import Postgres, PostgresConnection, PostgresRow

from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationAssistantTurn,
    ConversationTurn,
    ConversationUserTurn,
)
from draive.models import ModelContext

__all__ = ("PostgresConversationMemory",)


@final
class PostgresConversationMemory:
    """PostgreSQL-backed conversation memory factory.

    This utility exposes static helpers for schema migration and creating
    thread-scoped :class:`~draive.conversation.state.ConversationMemory`
    instances persisted in PostgreSQL.

    Examples
    --------
    ```python
    from uuid import uuid4

    from draive import ctx
    from draive.postgres.memory import PostgresConversationMemory

    async def bootstrap_memory() -> None:
        async with ctx.scope("conversation-memory"):
            await PostgresConversationMemory.migrate()
            memory = PostgresConversationMemory.prepare(thread=uuid4())
            await memory.recall()
    ```
    """

    @staticmethod
    async def migrate() -> None:
        """Create database structures required by conversation memory.

        This asynchronous method creates the `conversation_memory` table and
        its supporting index when they do not already exist.

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
            CREATE TABLE IF NOT EXISTS conversation_memory (
                thread_id TEXT NOT NULL,
                turn TEXT NOT NULL,
                identifier UUID NOT NULL,
                payload JSONB NOT NULL,
                created TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (thread_id, identifier),
                UNIQUE (thread_id, identifier)
            );

            CREATE INDEX IF NOT EXISTS conversation_memory_idx
                ON conversation_memory (thread_id, created DESC, identifier DESC);
            """
        )

    @staticmethod
    def prepare(
        *,
        thread: UUID | str,
    ) -> ConversationMemory:
        """Prepare thread-scoped conversation memory operations.

        Parameters
        ----------
        thread : UUID | str
            Conversation thread identifier used to isolate persisted turns.

        Returns
        -------
        ConversationMemory
            A configured conversation memory instance with fetch, recall, and
            remember handlers bound to the provided thread.

        Raises
        ------
        ValueError
            Raised by memory operations if pagination token validation fails.
        Exception
            Raised by memory operations when PostgreSQL interactions fail.
        """
        thread_id: str = str(thread)

        async def fetch(
            pagination: Pagination,
            **extra: Any,
        ) -> Paginated[ConversationTurn]:
            return await _fetch_turns(
                thread_id=thread_id,
                pagination=pagination,
            )

        async def recall(
            pagination: Pagination | None = None,
            **extra: Any,
        ) -> ModelContext:
            turns: Sequence[ConversationTurn]
            if pagination is None:
                turns = await _recall(
                    thread_id=thread_id,
                )

            else:
                turns = await _fetch_turns(
                    thread_id=thread_id,
                    pagination=pagination,
                )

            return tuple(chain.from_iterable(turn.to_model_context() for turn in turns))

        async def remember(
            turns: Sequence[ConversationTurn],
            **extra: Any,
        ) -> None:
            if not turns:
                return

            async with Postgres.acquire_connection() as connection:
                async with connection.transaction():
                    for turn in turns:
                        await connection.execute(
                            """
                            INSERT INTO
                                conversation_memory (
                                    thread_id,
                                    turn,
                                    identifier,
                                    payload,
                                    created
                                )

                            VALUES (
                                $1::TEXT,
                                $2::TEXT,
                                $3::UUID,
                                $4::JSONB,
                                $5::TIMESTAMPTZ
                            );
                            """,  # nosec: B608
                            thread_id,
                            turn.turn,
                            turn.identifier,
                            turn.to_json(),
                            turn.created,
                        )

            ctx.log_debug("...conversation memory persisted.")

        return ConversationMemory(
            fetching=fetch,
            recalling=recall,
            remembering=remember,
        )

    __slots__ = ()

    def __init__(self) -> NoReturn:
        raise RuntimeError("PostgresConversationMemory instantiation is forbidden")


def _turn_from_row(
    row: PostgresRow,
    /,
) -> ConversationTurn:
    payload: str = cast(str, row["payload"])
    parsed: Mapping[str, object] = cast(Mapping[str, object], json.loads(payload))

    turn_kind: object | None = parsed.get("turn")
    if turn_kind == "user":
        return ConversationUserTurn.from_json(payload)

    if turn_kind == "assistant":
        return ConversationAssistantTurn.from_json(payload)

    raise ValueError(f"Unsupported conversation turn payload: {turn_kind}")


async def _recall(
    *,
    thread_id: str,
) -> Sequence[ConversationTurn]:
    rows: Sequence[PostgresRow] = await Postgres.fetch(
        """
        SELECT
            payload::TEXT

        FROM
            conversation_memory

        WHERE
            thread_id = $1::TEXT

        ORDER BY
            created ASC,
            identifier ASC;
        """,  # nosec: B608
        thread_id,
    )

    return tuple(_turn_from_row(row) for row in rows)


async def _fetch_turns(
    *,
    thread_id: str,
    pagination: Pagination,
) -> Paginated[ConversationTurn]:
    if pagination.limit <= 0:
        return Paginated[ConversationTurn].of(
            (),
            pagination=pagination.with_token(None),
        )

    cursor: UUID | None
    if pagination.token is None:
        cursor = None

    elif isinstance(pagination.token, UUID):
        cursor = pagination.token

    elif isinstance(pagination.token, str):
        try:
            cursor = UUID(pagination.token)

        except ValueError as exc:
            raise ValueError("Invalid conversation memory pagination token") from exc

    else:
        raise ValueError("Invalid conversation memory pagination token")

    rows: Sequence[PostgresRow]
    if cursor is None:
        rows = await Postgres.fetch(
            """
            SELECT
                payload::TEXT,
                created,
                identifier

            FROM (
                SELECT
                    payload,
                    created,
                    identifier

                FROM
                    conversation_memory

                WHERE
                    thread_id = $1::TEXT

                ORDER BY
                    created DESC,
                    identifier DESC

                LIMIT $2::BIGINT
            ) AS recent_turns

            ORDER BY
                created ASC,
                identifier ASC;
            """,  # nosec: B608
            thread_id,
            pagination.limit,
        )

    else:
        rows = await Postgres.fetch(
            """
            WITH cursor AS (
                SELECT
                    created,
                    identifier

                FROM
                    conversation_memory

                WHERE
                    thread_id = $1::TEXT
                AND
                    identifier = $2::UUID
            )

            SELECT
                payload::TEXT,
                created,
                identifier

            FROM (
                SELECT
                    payload,
                    created,
                    identifier

                FROM
                    conversation_memory

                WHERE
                    thread_id = $1::TEXT
                AND (
                    created < (SELECT created FROM cursor)
                    OR (
                        created = (SELECT created FROM cursor)
                        AND identifier < (SELECT identifier FROM cursor)
                    )
                )

                ORDER BY
                    created DESC,
                    identifier DESC

                LIMIT $3::BIGINT
            ) AS recent_turns

            ORDER BY
                created ASC,
                identifier ASC;
            """,  # nosec: B608
            thread_id,
            cursor,
            pagination.limit,
        )

    next_token: UUID | None = None
    if len(rows) >= pagination.limit:
        next_token = rows[0].get_uuid("identifier", required=True)

    return Paginated[ConversationTurn].of(
        (_turn_from_row(row) for row in rows),
        pagination=pagination.with_token(next_token),
    )
