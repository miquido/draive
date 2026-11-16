import asyncio
from collections.abc import AsyncGenerator, Iterable, MutableSequence, Sequence
from typing import Any

import pytest
from haiway import Paginated, Pagination, ctx

from draive import (
    Conversation,
    ConversationAssistantTurn,
    ConversationEvent,
    ConversationMemory,
    ConversationTurn,
    ConversationUserTurn,
    GenerativeModel,
    ModelInput,
    ModelOutput,
    ModelOutputChunk,
    ModelReasoning,
    ModelSession,
    ModelSessionEvent,
    ModelSessionInputChunk,
    ModelSessionOutputChunk,
    ModelSessionScope,
    ModelToolRequest,
    ModelToolResponse,
    MultimodalContent,
    ProcessingEvent,
    RealtimeGenerativeModel,
    ResourceContent,
    Template,
    TemplatesRepository,
    TextContent,
    tool,
)
from draive.conversation.realtime.default import realtime_conversation_preparing
from draive.models import ModelContextElement, ModelTools
from draive.tools import Toolbox


def _text(content: Iterable[object]) -> str:
    return MultimodalContent.of(*content).to_str()  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_templated_message_keeps_multimodal_arguments() -> None:
    contexts: MutableSequence[Sequence[ModelContextElement]] = []

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any = "auto",
        **extra: Any,
    ) -> AsyncGenerator[ModelOutputChunk]:
        contexts.append(tuple(context))

        async def stream() -> AsyncGenerator[ModelOutputChunk]:
            yield TextContent.of("ok")

        return stream()

    image = ResourceContent.of(b"\x89PNG", mime_type="image/png")
    async with ctx.scope(
        "test",
        GenerativeModel(generating=generating),
        TemplatesRepository.volatile(look="Look: {%pic%} please"),
    ):
        _ = [
            chunk
            async for chunk in Conversation.completion(
                message=Template.of("look", arguments={"pic": image}),
            )
        ]

    first = contexts[0][0]
    assert isinstance(first, ModelInput)
    assert any(isinstance(part, ResourceContent) for part in first.content.parts)


def _tool_turn() -> ConversationAssistantTurn:
    request = ModelToolRequest.of(
        "call-1",
        tool="lookup",
        arguments={"name": "ada", "nested": {"a": [1, 2]}},
        handling="output",
        meta={"k": "v"},
    )
    response = ModelToolResponse.of(
        "call-1",
        tool="lookup",
        status="error",
        content="boom",
    )
    return ConversationAssistantTurn.of(
        MultimodalContent.of("before"),
        ModelReasoning.of("thinking", meta={"signature": "s"}),
        ConversationEvent.tool_request(request),
        ConversationEvent.tool_response(response),
        MultimodalContent.of("after"),
    )


def test_assistant_turn_with_tool_events_serializes_back_identically() -> None:
    turn = _tool_turn()
    restored = ConversationAssistantTurn.from_json(turn.to_json())

    assert restored == turn
    assert list(restored.to_model_context()) == list(turn.to_model_context())


def test_assistant_turn_context_alternates_output_input_output() -> None:
    context = list(_tool_turn().to_model_context())

    assert [type(element) for element in context] == [ModelOutput, ModelInput, ModelOutput]
    first = context[0]
    assert isinstance(first, ModelOutput)
    assert [type(block).__name__ for block in first.output] == [
        "MultimodalContent",
        "ModelReasoning",
        "ModelToolRequest",
    ]
    middle = context[1]
    assert isinstance(middle, ModelInput)
    assert [response.status for response in middle.tool_responses] == ["error"]


def test_assistant_turn_skips_tool_progress_events_in_context() -> None:
    turn = ConversationAssistantTurn.of(
        MultimodalContent.of("text"),
        ConversationEvent.tool_event(ProcessingEvent.of("progress")),
    )

    context = list(turn.to_model_context())

    assert len(context) == 1
    assert isinstance(context[0], ModelOutput)


def test_empty_turns_produce_no_assistant_context() -> None:
    assert list(ConversationAssistantTurn.of().to_model_context()) == []


# realtime


class _FakeRealtime:
    def __init__(self) -> None:
        self.output: MutableSequence[ModelSessionOutputChunk] = []
        self.written: MutableSequence[ModelSessionInputChunk] = []
        self.wakeup: asyncio.Event = asyncio.Event()
        self.closed: MutableSequence[type[BaseException] | None] = []

    def push(self, *chunks: ModelSessionOutputChunk) -> None:
        self.output.extend(chunks)
        self.wakeup.set()

    def model(self) -> RealtimeGenerativeModel:
        async def read() -> ModelSessionOutputChunk:
            while not self.output:
                self.wakeup.clear()
                await self.wakeup.wait()

            return self.output.pop(0)

        async def write(
            input: ModelSessionInputChunk,  # noqa: A002
        ) -> None:
            self.written.append(input)

        async def open_session() -> ModelSession:
            return ModelSession(reading=read, writing=write)

        async def close_session(
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: Any,
        ) -> None:
            self.closed.append(exc_type)

        def session_preparing(
            *,
            instructions: str,
            tools: ModelTools,
            context: Sequence[ModelContextElement],
            output: Any,
            **extra: Any,
        ) -> ModelSessionScope:
            return ModelSessionScope(opening=open_session, closing=close_session)

        return RealtimeGenerativeModel(session_preparing=session_preparing)


def _recording_memory(
    remembered: MutableSequence[ConversationTurn],
) -> ConversationMemory:
    async def fetch(
        pagination: Pagination,
        **extra: Any,
    ) -> Paginated[ConversationTurn]:
        return Paginated[ConversationTurn](
            tuple(remembered),
            pagination=pagination.with_token(None),
        )

    async def recall(
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Sequence[ModelContextElement]:
        return ()

    async def remember(
        turns: Sequence[ConversationTurn],
        **extra: Any,
    ) -> None:
        remembered.extend(turns)

    return ConversationMemory(fetching=fetch, recalling=recall, remembering=remember)


def _tool_request(identifier: str = "1") -> ModelToolRequest:
    return ModelToolRequest.of(identifier, tool="echo", arguments={"value": "v"})


@pytest.mark.asyncio
async def test_realtime_memory_keeps_assistant_output_in_order_across_tool_calls() -> None:
    fake = _FakeRealtime()
    remembered: MutableSequence[ConversationTurn] = []
    fake.push(
        ModelSessionEvent.turn_completed(
            ModelOutput.of(MultimodalContent.of("thinking"), _tool_request()),
        ),
        ModelSessionEvent.turn_completed(
            ModelInput.of(ModelToolResponse.of("1", tool="echo", content="v")),
        ),
        ModelSessionEvent.turn_completed(ModelOutput.of(MultimodalContent.of("final"))),
        TextContent.of("end"),
    )
    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.empty,
            memory=_recording_memory(remembered),
            output="text",
        )
        async with scope as session:
            async with asyncio.timeout(2):
                assert await session.read() == TextContent.of("end")

    assert len(remembered) == 1
    turn = remembered[0]
    assert isinstance(turn, ConversationAssistantTurn)
    # tool exchange is kept in order: output text, request, response, final output
    assert [
        element.event if isinstance(element, ConversationEvent) else element.to_str()
        for element in turn.content
    ] == ["thinking", "tool_request", "tool_response", "final"]
    assert [type(element).__name__ for element in turn.to_model_context()] == [
        "ModelOutput",
        "ModelInput",
        "ModelOutput",
    ]


@pytest.mark.asyncio
async def test_realtime_memory_remembers_user_then_assistant_turns() -> None:
    fake = _FakeRealtime()
    remembered: MutableSequence[ConversationTurn] = []
    fake.push(
        ModelSessionEvent.turn_completed(ModelInput.of(MultimodalContent.of("question"))),
        ModelSessionEvent.turn_completed(ModelOutput.of(MultimodalContent.of("answer"))),
        TextContent.of("end"),
    )
    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.empty,
            memory=_recording_memory(remembered),
            output="text",
        )
        async with scope as session:
            async with asyncio.timeout(2):
                assert await session.read() == TextContent.of("end")

    assert [type(turn) for turn in remembered] == [ConversationUserTurn, ConversationAssistantTurn]
    assert [_text(turn.content) for turn in remembered] == ["question", "answer"]


@pytest.mark.asyncio
async def test_realtime_tool_request_is_answered_through_the_session() -> None:
    fake = _FakeRealtime()

    @tool
    async def echo(value: str) -> str:
        return value

    fake.push(_tool_request())
    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.of(echo),
            memory=ConversationMemory.disabled,
            output="text",
        )
        async with scope as session:
            async with asyncio.timeout(2):
                event = await session.read()
                assert isinstance(event, ConversationEvent)
                assert event.event == "tool_request"

                await asyncio.sleep(0.05)  # let the tool finish
                # the provider speaks again once it received the tool result
                fake.push(TextContent.of("after tool"), TextContent.of("done"))
                chunks = [await session.read() for _ in range(3)]

    assert any(isinstance(chunk, TextContent) for chunk in chunks)
    assert any(
        isinstance(chunk, ConversationEvent) and chunk.event == "tool_response" for chunk in chunks
    )
    responses = [chunk for chunk in fake.written if isinstance(chunk, ModelToolResponse)]
    assert [(response.identifier, response.status) for response in responses] == [("1", "success")]
    assert responses[0].content.to_str() == "v"


@pytest.mark.asyncio
async def test_realtime_tool_events_are_delivered_while_provider_is_silent() -> None:
    fake = _FakeRealtime()

    @tool
    async def echo(value: str) -> str:
        return value

    fake.push(_tool_request())
    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.of(echo),
            memory=ConversationMemory.disabled,
            output="text",
        )
        async with scope as session:
            async with asyncio.timeout(2):
                first = await session.read()
                assert isinstance(first, ConversationEvent)
                assert first.event == "tool_request"

            async with asyncio.timeout(1):
                second = await session.read()

            assert isinstance(second, ConversationEvent)
            assert second.event == "tool_response"


@pytest.mark.asyncio
async def test_realtime_read_without_running_tools_spawns_no_waiter_task() -> None:
    fake = _FakeRealtime()
    fake.push(TextContent.of("a"), TextContent.of("b"), TextContent.of("c"))
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    coroutine_names: list[str] = []

    def factory(
        loop: asyncio.AbstractEventLoop,
        coro: Any,
        **kwargs: Any,
    ) -> asyncio.Future[Any]:
        coroutine_names.append(getattr(coro, "__qualname__", ""))
        return asyncio.Task(coro, loop=loop, **kwargs)

    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.empty,
            memory=ConversationMemory.disabled,
            output="text",
        )
        async with scope as session:
            loop.set_task_factory(factory)  # pyright: ignore[reportArgumentType]
            try:
                async with asyncio.timeout(2):
                    chunks = [await session.read() for _ in range(3)]

            finally:
                loop.set_task_factory(previous_factory)

    assert [chunk.to_str() for chunk in chunks if isinstance(chunk, TextContent)] == ["a", "b", "c"]
    # a single provider read per chunk and no tools output waiters
    assert len(coroutine_names) == 3
    assert not any("Event.wait" in name for name in coroutine_names)


@pytest.mark.asyncio
async def test_realtime_close_leaves_no_pending_tasks() -> None:
    fake = _FakeRealtime()

    @tool
    async def echo(value: str) -> str:
        return value

    fake.push(_tool_request())
    before = asyncio.all_tasks()
    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.of(echo),
            memory=ConversationMemory.disabled,
            output="text",
        )
        async with scope as session:
            async with asyncio.timeout(2):
                await session.read()  # tool request
                await session.read()  # tool response, provider stays silent
                # leaves a provider read pending during close
                reading = asyncio.ensure_future(session.read())
                await asyncio.sleep(0.01)
                reading.cancel()

    await asyncio.sleep(0)
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.asyncio
async def test_realtime_session_closes_with_the_failure_that_ended_it() -> None:
    fake = _FakeRealtime()
    async with ctx.scope("test", fake.model()):
        scope = realtime_conversation_preparing(
            instructions="",
            toolbox=Toolbox.empty,
            memory=ConversationMemory.disabled,
            output="text",
        )
        with pytest.raises(RuntimeError):
            async with scope:
                raise RuntimeError("failure")

    assert fake.closed == [RuntimeError]
