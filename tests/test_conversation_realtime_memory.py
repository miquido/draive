from collections.abc import Iterable, Sequence
from typing import Any, Literal

import pytest
from haiway import Paginated, Pagination, ctx

from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationTurn,
    ConversationUserTurn,
)
from draive.models import (
    ModelContextElement,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelSession,
    ModelSessionEvent,
    ModelSessionInputChunk,
    ModelSessionOutputChunk,
    ModelSessionOutputSelection,
    ModelSessionScope,
    ModelTools,
    RealtimeGenerativeModel,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.tools import Toolbox


def _memory_completed_event(
    *,
    identifier: str,
    role: Literal["user", "assistant"],
    text: str,
) -> ModelSessionEvent:
    match role:
        case "user":
            return ModelSessionEvent.turn_completed(
                ModelInput.of(
                    MultimodalContent.of(TextContent.of(text)),
                    meta={"identifier": identifier},
                )
            )

        case "assistant":
            return ModelSessionEvent.turn_completed(
                ModelOutput.of(
                    MultimodalContent.of(TextContent.of(text)),
                    meta={"identifier": identifier},
                )
            )


@pytest.mark.asyncio
async def test_realtime_memory_completed_event_is_remembered_and_not_forwarded() -> None:
    remembered_turns: list[ConversationTurn] = []
    output_chunks: list[ModelSessionOutputChunk] = [
        _memory_completed_event(
            identifier="00000000-0000-0000-0000-000000000001",
            role="user",
            text="hello from mic",
        ),
        TextContent.of("assistant output"),
    ]

    async def read() -> ModelSessionOutputChunk:
        return output_chunks.pop(0)

    async def write(
        input: ModelSessionInputChunk,  # noqa: A002
    ) -> None:
        _ = input

    async def open_session() -> ModelSession:
        return ModelSession(
            reading=read,
            writing=write,
        )

    async def close_session(
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)

    def session_prepare(
        *,
        instructions: ModelInstructions,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: ModelSessionOutputSelection,
        **extra: Any,
    ) -> ModelSessionScope:
        _ = (instructions, tools, context, output, extra)
        return ModelSessionScope(opening=open_session, closing=close_session)

    async def fetch(
        pagination: Pagination,
        **extra: Any,
    ) -> Paginated[ConversationTurn]:
        _ = extra
        return Paginated[ConversationTurn].of(
            remembered_turns,
            pagination=pagination.with_token(None),
        )

    async def recall(
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Sequence[ModelContextElement]:
        _ = (pagination, extra)
        return ()

    async def remember(
        turns: Iterable[ConversationTurn],
        **extra: Any,
    ) -> None:
        _ = extra
        remembered_turns.extend(turns)

    memory = ConversationMemory(
        fetching=fetch,
        recalling=recall,
        remembering=remember,
    )

    async with ctx.scope(
        "test",
        RealtimeGenerativeModel(session_preparing=session_prepare),
    ):
        session_scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.empty,
            memory=memory,
            output="text",
        )

        async with session_scope as session:
            chunk = await session.read()

    assert isinstance(chunk, TextContent)
    assert chunk.text == "assistant output"
    assert len(remembered_turns) == 1
    assert isinstance(remembered_turns[0], ConversationUserTurn)
    assert remembered_turns[0].content[0].to_str() == "hello from mic"
