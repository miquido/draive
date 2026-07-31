from collections.abc import AsyncIterable, Iterable, Sequence
from types import TracebackType
from typing import Any
from uuid import uuid4

import pytest
from haiway import State, ctx

from draive import Agent
from draive.agents import AgentException, AgentMemory, AgentThread
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelOutput,
    ModelOutputChunk,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import MultimodalContent, MultimodalContentPart, TextContent
from draive.steps import StepState, step
from draive.tools import Tool, tool
from draive.utils import ProcessingEvent


async def _stream_of(*chunks: ModelOutputChunk) -> AsyncIterable[ModelOutputChunk]:
    for chunk in chunks:
        yield chunk


def _text_of(chunks: Iterable[MultimodalContentPart | ProcessingEvent]) -> str:
    return MultimodalContent.of(
        *(chunk for chunk in chunks if not isinstance(chunk, ProcessingEvent))
    ).to_str()


class _MarkerState(State):
    label: str


def _thread(agent_uri: str = "agent://agent") -> AgentThread:
    return AgentThread.of(uuid4(), agent_uri=agent_uri)


async def _noop_remembering(
    thread: AgentThread,
    context: ModelContext,
    **extra: object,
) -> None:
    _ = (thread, context, extra)


@pytest.mark.asyncio
async def test_agent_memory_with_ctx_injects_state_into_recall_and_remember() -> None:
    captured: dict[str, str] = {}

    async def preparing(
        thread: AgentThread,
        instructions: str,
        **extra: object,
    ) -> None:
        _ = (thread, instructions, extra)
        captured["prepare_marker"] = ctx.state(_MarkerState).label

    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        captured["recall_marker"] = ctx.state(_MarkerState).label
        return context

    async def remembering(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> None:
        _ = (thread, context, extra)
        captured["remember_marker"] = ctx.state(_MarkerState).label

    memory = AgentMemory(
        recalling=recalling,
        remembering=remembering,
        preparing=preparing,
    ).with_ctx(_MarkerState(label="injected"))

    thread = _thread()
    context: ModelContext = (ModelInput.of(MultimodalContent.of("hi")),)

    async with ctx.scope("test"):
        await memory.prepare(thread=thread, instructions="instructions")
        recalled = await memory.recall(thread=thread, context=context)
        await memory.remember(thread=thread, context=context)

    assert recalled == context
    assert captured["prepare_marker"] == "injected"
    assert captured["recall_marker"] == "injected"
    assert captured["remember_marker"] == "injected"


@pytest.mark.asyncio
async def test_agent_memory_with_ctx_applies_to_steps_too() -> None:
    captured: dict[str, str] = {}

    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        captured["marker"] = ctx.state(_MarkerState).label
        return context

    memory = AgentMemory(recalling=recalling, remembering=_noop_remembering).with_ctx(
        _MarkerState(label="from-step")
    )

    state = StepState.of((ModelInput.of(MultimodalContent.of("hi")),))

    async with ctx.scope("test", _thread()):
        await memory.recall_step().process(state)

    assert captured["marker"] == "from-step"


@pytest.mark.asyncio
async def test_agent_memory_prepare_passes_thread_and_instructions() -> None:
    captured: dict[str, object] = {}

    async def preparing(
        thread: AgentThread,
        instructions: str,
        **extra: object,
    ) -> None:
        _ = extra
        captured["thread"] = thread
        captured["instructions"] = instructions

    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        return context

    memory = AgentMemory(
        recalling=recalling,
        remembering=_noop_remembering,
        preparing=preparing,
    )

    thread = _thread()

    async with ctx.scope("test"):
        assert await memory.prepare(thread=thread, instructions="agent instructions") is None

    assert captured["thread"] == thread
    assert captured["instructions"] == "agent instructions"


@pytest.mark.asyncio
async def test_agent_generative_extends_toolbox_with_prepared_tools() -> None:
    tool_calls: list[str] = []
    offered_tools: list[tuple[str, ...]] = []

    @tool(name="agent_tool")
    async def agent_tool() -> str:
        return "agent"

    async def preparing(
        thread: AgentThread,
        instructions: str,
        **extra: object,
    ) -> Sequence[Tool]:
        _ = (instructions, extra)

        @tool(name="memory_recall")
        async def memory_recall(topic: str) -> str:  # closes over prepared turn state
            tool_calls.append(f"{topic}@{thread.agent_uri}")
            return f"details about {topic}"

        return (memory_recall,)

    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        return context

    iteration: int = 0

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        _ = (instructions, context, output, extra)
        nonlocal iteration
        offered_tools.append(tuple(available.name for available in tools.specification))
        iteration += 1
        if iteration > 1:
            return _stream_of(TextContent.of("final"))

        return _stream_of(
            ModelToolRequest.of(
                "call1",
                tool="memory_recall",
                arguments={"topic": "budget"},
            )
        )

    agent = Agent.generative(
        "helper",
        instructions="be helpful",
        tools=[agent_tool],
        memory=AgentMemory(
            recalling=recalling,
            remembering=_noop_remembering,
            preparing=preparing,
        ),
    )

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        chunks = [chunk async for chunk in agent.call(input="hi")]

    # prepared tools stay available through all iterations of the turn
    assert offered_tools == [
        ("agent_tool", "memory_recall"),
        ("agent_tool", "memory_recall"),
    ]
    assert tool_calls == [f"budget@{agent.identity.uri}"]
    assert _text_of(chunks) == "final"


@pytest.mark.parametrize("prepared", [None, (), []])
@pytest.mark.asyncio
async def test_agent_generative_keeps_toolbox_without_prepared_tools(
    prepared: Sequence[Tool] | None,
) -> None:
    offered_tools: list[tuple[str, ...]] = []

    @tool(name="agent_tool")
    async def agent_tool() -> str:
        return "agent"

    async def preparing(
        thread: AgentThread,
        instructions: str,
        **extra: object,
    ) -> Sequence[Tool] | None:
        _ = (thread, instructions, extra)
        return prepared

    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        return context

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        _ = (instructions, context, output, extra)
        offered_tools.append(tuple(available.name for available in tools.specification))
        return _stream_of(TextContent.of("final"))

    agent = Agent.generative(
        "helper",
        instructions="be helpful",
        tools=[agent_tool],
        memory=AgentMemory(
            recalling=recalling,
            remembering=_noop_remembering,
            preparing=preparing,
        ),
    )

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        chunks = [chunk async for chunk in agent.call(input="hi")]

    assert offered_tools == [("agent_tool",)]
    assert _text_of(chunks) == "final"


@pytest.mark.asyncio
async def test_agent_memory_prepare_defaults_to_noop() -> None:
    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        return context

    memory = AgentMemory(recalling=recalling, remembering=_noop_remembering)

    # no preparing provided - prepare completes without effect
    await memory.prepare(thread=_thread(), instructions="anything")


@pytest.mark.asyncio
async def test_agent_memory_with_ctx_opens_and_closes_disposables() -> None:
    events: list[str] = []

    class _FakeDisposable:
        async def __aenter__(self) -> _MarkerState:
            events.append("enter")
            return _MarkerState(label="disposed")

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            events.append("exit")

    async def recalling(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        events.append(f"recall:{ctx.state(_MarkerState).label}")
        return context

    memory = AgentMemory(recalling=recalling, remembering=_noop_remembering).with_ctx(
        disposables=(_FakeDisposable(),)
    )

    async with ctx.scope("test"):
        await memory.recall(
            thread=_thread(),
            context=(ModelInput.of(MultimodalContent.of("hi")),),
        )

    assert events == ["enter", "recall:disposed", "exit"]


def test_agent_memory_with_ctx_without_arguments_returns_self() -> None:
    memory = AgentMemory.volatile()

    assert memory.with_ctx() is memory


@pytest.mark.asyncio
async def test_agent_memory_volatile_accumulates_context_across_turns() -> None:
    memory = AgentMemory.volatile()
    thread = _thread()

    first_input = ModelInput.of(MultimodalContent.of("first"))
    first_recalled = await memory.recall(thread=thread, context=(first_input,))
    assert first_recalled == (first_input,)

    first_output = ModelOutput.of(MultimodalContent.of("second"))
    await memory.remember(thread=thread, context=(*first_recalled, first_output))

    second_input = ModelInput.of(MultimodalContent.of("third"))
    second_recalled = await memory.recall(thread=thread, context=(second_input,))
    assert len(second_recalled) == 3
    assert second_recalled[0].content.to_str() == "first"
    assert second_recalled[1].content.to_str() == "second"
    assert second_recalled[2] is second_input

    second_output = ModelOutput.of(MultimodalContent.of("fourth"))
    await memory.remember(thread=thread, context=(*second_recalled, second_output))

    third_recalled = await memory.recall(
        thread=thread,
        context=(ModelInput.of(MultimodalContent.of("fifth")),),
    )
    assert [element.content.to_str() for element in third_recalled] == [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
    ]


def test_agent_memory_volatile_accepts_initial_keyword() -> None:
    initial = ModelInput.of(MultimodalContent.of("initial"))
    memory = AgentMemory.volatile(initial=(initial,))

    assert isinstance(memory, AgentMemory)


@pytest.mark.asyncio
async def test_agent_memory_volatile_snapshots_initial_context() -> None:
    initial = [ModelInput.of(MultimodalContent.of("initial"))]
    memory = AgentMemory.volatile(initial=initial)
    initial.clear()

    incoming = ModelInput.of(MultimodalContent.of("incoming"))
    recalled = await memory.recall(
        thread=_thread(),
        context=(incoming,),
    )

    assert [element.content.to_str() for element in recalled] == ["initial", "incoming"]


@pytest.mark.asyncio
async def test_agent_memory_volatile_stores_remembered_context_as_snapshot() -> None:
    memory = AgentMemory.volatile()
    thread = _thread()

    await memory.remember(
        thread=thread,
        context=(
            ModelInput.of(MultimodalContent.of("original")),
            ModelOutput.of(MultimodalContent.of("history")),
        ),
    )

    # e.g. a compaction step replaced the whole context with a summary;
    # the remembered context becomes the next recalled one, as provided
    summary_input = ModelInput.of(MultimodalContent.of("summary of prior conversation"))
    await memory.remember(thread=thread, context=(summary_input,))

    incoming = ModelInput.of(MultimodalContent.of("next"))
    assert await memory.recall(thread=thread, context=(incoming,)) == (summary_input, incoming)


@pytest.mark.asyncio
async def test_agent_memory_volatile_isolates_agents_within_thread() -> None:
    memory = AgentMemory.volatile()
    identifier = uuid4()
    first_thread = AgentThread.of(identifier, agent_uri="agent://first")
    second_thread = AgentThread.of(identifier, agent_uri="agent://second")

    await memory.remember(
        thread=first_thread,
        context=(ModelInput.of(MultimodalContent.of("first agent history")),),
    )

    # the second agent shares the thread yet recalls no foreign context
    incoming = ModelInput.of(MultimodalContent.of("hello"))
    assert await memory.recall(thread=second_thread, context=(incoming,)) == (incoming,)

    recalled = await memory.recall(thread=first_thread, context=(incoming,))
    assert [element.content.to_str() for element in recalled] == [
        "first agent history",
        "hello",
    ]


@pytest.mark.asyncio
async def test_agent_memory_volatile_prepare_refreshes_entry_recency() -> None:
    memory = AgentMemory.volatile(threads_limit=2)
    first_thread = _thread()
    second_thread = _thread()

    for thread in (first_thread, second_thread):
        await memory.remember(
            thread=thread,
            context=(ModelInput.of(MultimodalContent.of("stored")),),
        )

    # preparing the first entry marks it as recently used...
    await memory.prepare(thread=first_thread, instructions="instructions")

    # ...so remembering a new entry evicts the second one instead
    await memory.remember(
        thread=_thread(),
        context=(ModelInput.of(MultimodalContent.of("stored")),),
    )
    first_input = ModelInput.of(MultimodalContent.of("next"))
    assert len(await memory.recall(thread=first_thread, context=(first_input,))) == 2
    second_input = ModelInput.of(MultimodalContent.of("fresh"))
    assert await memory.recall(thread=second_thread, context=(second_input,)) == (second_input,)


@pytest.mark.asyncio
async def test_agent_steps_with_memory_persist_context_across_calls() -> None:
    memory = AgentMemory.volatile()

    @step
    async def reply(
        state: StepState,
    ) -> StepState:
        return state.appending_context(ModelOutput.of(MultimodalContent.of("reply")))

    agent = Agent.steps(
        memory.recall_step(),
        reply,
        memory.remember_step(),
        agent="worker",
    )
    thread = uuid4()

    async with ctx.scope("test.agent.memory"):
        _ = [chunk async for chunk in agent.call(thread=thread, input="first")]
        _ = [chunk async for chunk in agent.call(thread=thread, input="second")]

    # context is stored under the executing agent's URI stamped by respond
    stored = await memory.recall(
        thread=AgentThread.of(thread, agent_uri=agent.identity.uri),
        context=(),
    )
    assert [element.content.to_str() for element in stored] == [
        "first",
        "reply",
        "second",
        "reply",
    ]

    # a different agent URI on the same thread sees no stored context
    assert (
        await memory.recall(
            thread=AgentThread.of(thread, agent_uri="agent://other"),
            context=(),
        )
        == ()
    )


@pytest.mark.asyncio
async def test_agent_memory_steps_require_agent_thread_in_scope() -> None:
    memory = AgentMemory.volatile()
    state = StepState.of(())

    async with ctx.scope("test"):  # no AgentThread bound in scope
        with pytest.raises(AgentException):
            await memory.recall_step().process(state)

        with pytest.raises(AgentException):
            await memory.remember_step().process(state)
