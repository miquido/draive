import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import chain
from typing import Any, cast
from uuid import UUID

from haiway import Paginated, Pagination, ctx

from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationAssistantTurn,
    ConversationTurn,
    ConversationUserTurn,
)
from draive.models import ModelContext
from draive.surreal.state import Surreal
from draive.surreal.types import SurrealObject

__all__ = ("SurrealConversationMemory",)


def SurrealConversationMemory(
    *,
    thread: UUID | str,
) -> ConversationMemory:
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
            turns = await _recall(thread_id=thread_id)

        else:
            turns = (await _fetch_turns(thread_id=thread_id, pagination=pagination)).items

        return tuple(chain.from_iterable(turn.to_model_context() for turn in turns))

    async def remember(
        turns: Sequence[ConversationTurn],
        **extra: Any,
    ) -> None:
        _ = extra
        if not turns:
            return

        for turn in turns:
            await Surreal.execute(
                """
                CREATE conversation_memory SET
                    thread_id = $thread_id,
                    turn = $turn,
                    identifier = $identifier,
                    payload = $payload,
                    created = $created;
                """,
                thread_id=thread_id,
                turn=turn.turn,
                identifier=str(turn.identifier),
                payload=turn.to_json(),
                created=turn.created,
            )

        ctx.log_debug("...conversation memory persisted in SurrealDB.")

    return ConversationMemory(
        fetching=fetch,
        recalling=recall,
        remembering=remember,
    )


def _turn_from_record(
    record: SurrealObject,
    /,
) -> ConversationTurn:
    payload: str = cast(str, record["payload"])
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
    rows: Sequence[SurrealObject] = await Surreal.execute(
        """
        SELECT
            payload,
            created,
            identifier
        FROM
            conversation_memory
        WHERE
            thread_id = $thread_id
        ORDER BY
            created ASC,
            identifier ASC;
        """,
        thread_id=thread_id,
    )

    return tuple(_turn_from_record(row) for row in rows)


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

    cursor: str | None = _memory_pagination_token(pagination)
    fetch_limit: int = pagination.limit + 1
    rows: Sequence[SurrealObject]
    if cursor is None:
        rows = await Surreal.execute(
            """
            SELECT
                payload,
                created,
                identifier
            FROM
                conversation_memory
            WHERE
                thread_id = $thread_id
            ORDER BY
                created DESC,
                identifier DESC
            LIMIT $limit;
            """,
            thread_id=thread_id,
            limit=fetch_limit,
        )

    else:
        cursor_rows: Sequence[SurrealObject] = await Surreal.execute(
            """
            SELECT
                created,
                identifier
            FROM
                conversation_memory
            WHERE
                thread_id = $thread_id
            AND
                identifier = $cursor
            LIMIT 1;
            """,
            thread_id=thread_id,
            cursor=cursor,
        )
        if not cursor_rows:
            return Paginated[ConversationTurn].of(
                (),
                pagination=pagination.with_token(None),
            )

        cursor_row: SurrealObject = cursor_rows[0]
        cursor_created: datetime = cast(datetime, cursor_row["created"])
        cursor_identifier: str = cast(str, cursor_row["identifier"])

        rows = await Surreal.execute(
            """
            SELECT
                payload,
                created,
                identifier
            FROM
                conversation_memory
            WHERE
                thread_id = $thread_id
            AND (
                created < $cursor_created
                OR (
                    created = $cursor_created
                    AND identifier < $cursor_identifier
                )
            )
            ORDER BY
                created DESC,
                identifier DESC
            LIMIT $limit;
            """,
            thread_id=thread_id,
            cursor_created=cursor_created,
            cursor_identifier=cursor_identifier,
            limit=fetch_limit,
        )

    page_rows: Sequence[SurrealObject] = rows[: pagination.limit]
    next_token: str | None = None
    if len(rows) > pagination.limit and page_rows:
        next_token = f"conversation_memory:cursor:{cast(str, page_rows[-1]['identifier'])}"

    turns: Sequence[ConversationTurn] = tuple(_turn_from_record(row) for row in reversed(page_rows))
    return Paginated[ConversationTurn].of(
        turns,
        pagination=pagination.with_token(next_token),
    )


def _memory_pagination_token(
    pagination: Pagination,
    /,
) -> str | None:
    if pagination.token is None:
        return None

    if isinstance(pagination.token, UUID):
        return str(pagination.token)

    if isinstance(pagination.token, str):
        if pagination.token.startswith("conversation_memory:cursor:"):
            cursor: str = pagination.token.removeprefix("conversation_memory:cursor:")
            if cursor:
                return cursor

            raise ValueError("Invalid SurrealDB conversation memory pagination token")

        try:
            offset: int = max(int(pagination.token), 0)
        except ValueError as exc:
            raise ValueError("Invalid SurrealDB conversation memory pagination token") from exc

        if offset == 0:
            return None

        raise ValueError(
            "SurrealDB conversation memory no longer supports non-zero offset pagination tokens"
        )

    raise ValueError("Invalid SurrealDB conversation memory pagination token")
