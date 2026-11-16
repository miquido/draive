import asyncio
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
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    RealtimeGenerativeModel,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.tools import Toolbox, tool


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
        return Paginated[ConversationTurn](
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


class _Harness:
    def __init__(self, chunks: Sequence[ModelSessionOutputChunk]) -> None:
        self.chunks: list[ModelSessionOutputChunk] = list(chunks)
        self.written: list[ModelSessionInputChunk] = []
        self.remembered: list[ConversationTurn] = []
        self.tail: asyncio.Event = asyncio.Event()

    def memory(self) -> ConversationMemory:
        async def fetch(pagination: Pagination, **extra: Any) -> Paginated[ConversationTurn]:
            _ = extra
            return Paginated[ConversationTurn](
                self.remembered, pagination=pagination.with_token(None)
            )

        async def recall(
            pagination: Pagination | None = None, **extra: Any
        ) -> Sequence[ModelContextElement]:
            _ = (pagination, extra)
            return ()

        async def remember(turns: Iterable[ConversationTurn], **extra: Any) -> None:
            _ = extra
            self.remembered.extend(turns)

        return ConversationMemory(fetching=fetch, recalling=recall, remembering=remember)

    def model(self) -> RealtimeGenerativeModel:
        async def read() -> ModelSessionOutputChunk:
            if self.chunks:
                return self.chunks.pop(0)

            await self.tail.wait()  # nothing more from the provider
            raise asyncio.CancelledError

        async def write(input: ModelSessionInputChunk) -> None:  # noqa: A002
            self.written.append(input)

        async def open_session() -> ModelSession:
            return ModelSession(reading=read, writing=write)

        async def close_session(
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: Any,
        ) -> None:
            _ = (exc_type, exc_val, exc_tb)

        def prepare(**extra: Any) -> ModelSessionScope:
            _ = extra
            return ModelSessionScope(opening=open_session, closing=close_session)

        return RealtimeGenerativeModel(session_preparing=prepare)


def _text_turn(role: Literal["user", "assistant"], text: str) -> ModelSessionEvent:
    return _memory_completed_event(identifier="id", role=role, text=text)


async def _drain(session: Any, count: int) -> None:
    for _ in range(count):
        await asyncio.wait_for(session.read(), 1)


@pytest.mark.asyncio
async def test_realtime_interrupted_assistant_turn_does_not_attach_to_next_one() -> None:
    harness = _Harness(
        [
            ModelSessionEvent.turn_completed(
                ModelOutput.of(
                    ModelToolRequest.of("t1", tool="lookup"),
                    MultimodalContent.of("let me check"),
                )
            ),
            ModelSessionEvent.turn_completed(
                ModelInput.of(
                    ModelToolResponse(
                        identifier="t1",
                        tool="lookup",
                        status="success",
                        content=MultimodalContent.of("tool text"),
                    )
                )
            ),
            _text_turn("user", "never mind"),
            _text_turn("assistant", "sure"),
            TextContent.of("end"),
        ]
    )
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="", toolbox=Toolbox.empty, memory=harness.memory(), output="text"
        ) as session:
            await _drain(session, 1)

    kinds = [turn.turn for turn in harness.remembered]
    assert kinds == ["assistant", "user", "assistant"]
    assert harness.remembered[2].content[0].to_str() == "sure"
    assert "never mind" not in harness.remembered[2].content[0].to_str()


@pytest.mark.asyncio
async def test_realtime_empty_turns_are_not_remembered() -> None:
    harness = _Harness(
        [
            _text_turn("user", "hello"),
            ModelSessionEvent.turn_completed(ModelOutput.of()),
            TextContent.of("end"),
        ]
    )
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="", toolbox=Toolbox.empty, memory=harness.memory(), output="text"
        ) as session:
            await _drain(session, 1)

    assert [turn.turn for turn in harness.remembered] == ["user"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "cancel"])
async def test_realtime_failed_tool_handler_still_answers_the_request(failure: str) -> None:
    @tool
    async def broken() -> str:
        if failure == "cancel":
            raise asyncio.CancelledError

        raise RuntimeError("boom")

    harness = _Harness([ModelToolRequest.of("t1", tool="broken"), TextContent.of("end")])
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.of(broken),
            memory=harness.memory(),
            output="text",
        ) as session:
            await _drain(session, 3)
            await asyncio.sleep(0.05)

    responses = [chunk for chunk in harness.written if isinstance(chunk, ModelToolResponse)]
    assert [(response.identifier, response.status) for response in responses] == [("t1", "error")]


def _tool_response(identifier: str, *, status: Literal["success", "error"] = "success") -> Any:
    return ModelToolResponse(
        identifier=identifier,
        tool="lookup",
        status=status,
        content=MultimodalContent.of("tool text"),
    )


def _replayed_context(turns: Sequence[ConversationTurn]) -> list[ModelContextElement]:
    return [element for turn in turns for element in turn.to_model_context()]


def _kinds(context: Sequence[ModelContextElement]) -> list[str]:
    return [element.kind for element in context]


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_realtime_tool_exchange_is_remembered_and_replayed(parallel: bool) -> None:
    identifiers = ("t1", "t2") if parallel else ("t1",)
    harness = _Harness(
        [
            ModelSessionEvent.turn_completed(
                ModelOutput.of(
                    MultimodalContent.of("let me check"),
                    *(ModelToolRequest.of(identifier, tool="lookup") for identifier in identifiers),
                )
            ),
            ModelSessionEvent.turn_completed(
                ModelInput.of(*(_tool_response(identifier) for identifier in identifiers))
            ),
            _text_turn("assistant", "done"),
            TextContent.of("end"),
        ]
    )
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="", toolbox=Toolbox.empty, memory=harness.memory(), output="text"
        ) as session:
            await _drain(session, 1)

    assert [turn.turn for turn in harness.remembered] == ["assistant"]
    context = _replayed_context(harness.remembered)
    assert _kinds(context) == ["output", "input", "output"]
    first, second, last = context
    assert isinstance(first, ModelOutput)
    assert isinstance(second, ModelInput)
    assert isinstance(last, ModelOutput)
    assert [request.identifier for request in first.tool_requests] == list(identifiers)
    assert first.content.to_str() == "let me check"
    assert [response.identifier for response in second.tool_responses] == list(identifiers)
    assert last.content.to_str() == "done"


@pytest.mark.asyncio
async def test_realtime_tool_error_response_is_remembered_when_provider_does_not_report_it() -> (
    None
):
    @tool
    async def broken() -> str:
        raise RuntimeError("boom")

    harness = _Harness(
        [
            ModelToolRequest.of("t1", tool="broken"),
            ModelSessionEvent.turn_completed(
                ModelOutput.of(ModelToolRequest.of("t1", tool="broken"))
            ),
            TextContent.of("middle"),
        ]
    )
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.of(broken),
            memory=harness.memory(),
            output="text",
        ) as session:
            await _drain(session, 3)
            await asyncio.sleep(0.05)
            harness.chunks.extend([_text_turn("assistant", "sorry"), TextContent.of("end")])
            await _drain(session, 1)

    context = _replayed_context(harness.remembered)
    assert _kinds(context) == ["output", "input", "output"]
    assert isinstance(context[1], ModelInput)
    assert [(r.identifier, r.status) for r in context[1].tool_responses] == [("t1", "error")]


@pytest.mark.asyncio
async def test_realtime_interrupted_turn_drops_unanswered_tool_requests() -> None:
    harness = _Harness(
        [
            ModelSessionEvent.turn_completed(
                ModelOutput.of(
                    MultimodalContent.of("checking"),
                    ModelToolRequest.of("t1", tool="lookup"),
                    ModelToolRequest.of("t2", tool="lookup"),
                )
            ),
            ModelSessionEvent.turn_completed(ModelInput.of(_tool_response("t1"))),
            _text_turn("user", "never mind"),
            TextContent.of("end"),
        ]
    )
    async with ctx.scope("test", harness.model()):
        async with realtime_conversation_preparing(
            instructions="", toolbox=Toolbox.empty, memory=harness.memory(), output="text"
        ) as session:
            await _drain(session, 1)

    assert [turn.turn for turn in harness.remembered] == ["assistant", "user"]
    context = _replayed_context(harness.remembered)
    first = context[0]
    assert isinstance(first, ModelOutput)
    assert [request.identifier for request in first.tool_requests] == ["t1"]
    assert _kinds(context) == ["output", "input", "input"]
