from types import TracebackType
from uuid import uuid4

import pytest
from haiway import State, ctx

from draive.agents import AgentMemory, AgentThread
from draive.models import ModelContext, ModelInput, ModelOutput
from draive.multimodal import MultimodalContent
from draive.steps import StepState


class _MarkerState(State):
    label: str


async def _noop_remembering(
    thread: AgentThread,
    context: ModelContext,
    **extra: object,
) -> None:
    _ = (thread, context, extra)


@pytest.mark.asyncio
async def test_agent_memory_with_ctx_injects_state_into_recall_and_remember() -> None:
    captured: dict[str, str] = {}

    async def recalling(
        thread: AgentThread,
        input: ModelInput,  # noqa: A002
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        captured["recall_marker"] = ctx.state(_MarkerState).label
        return (input,)

    async def remembering(
        thread: AgentThread,
        context: ModelContext,
        **extra: object,
    ) -> None:
        _ = (thread, context, extra)
        captured["remember_marker"] = ctx.state(_MarkerState).label

    memory = AgentMemory(recalling=recalling, remembering=remembering).with_ctx(
        _MarkerState(label="injected")
    )

    thread = AgentThread.of(uuid4())
    context: ModelContext = (ModelInput.of(MultimodalContent.of("hi")),)

    async with ctx.scope("test"):
        recalled = await memory.recall(thread=thread, input=context[0])
        await memory.remember(thread=thread, context=context)

    assert recalled == context
    assert captured["recall_marker"] == "injected"
    assert captured["remember_marker"] == "injected"


@pytest.mark.asyncio
async def test_agent_memory_with_ctx_applies_to_step_properties_too() -> None:
    captured: dict[str, str] = {}

    async def recalling(
        thread: AgentThread,
        input: ModelInput,  # noqa: A002
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        captured["marker"] = ctx.state(_MarkerState).label
        return (input,)

    memory = AgentMemory(recalling=recalling, remembering=_noop_remembering).with_ctx(
        _MarkerState(label="from-step")
    )

    thread = AgentThread.of(uuid4())
    model_input = ModelInput.of(MultimodalContent.of("hi"))
    state = StepState.of(())

    async with ctx.scope("test", thread):
        await memory.recall_step(input=model_input).process(state)

    assert captured["marker"] == "from-step"


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
        input: ModelInput,  # noqa: A002
        **extra: object,
    ) -> ModelContext:
        _ = (thread, extra)
        events.append(f"recall:{ctx.state(_MarkerState).label}")
        return (input,)

    memory = AgentMemory(recalling=recalling, remembering=_noop_remembering).with_ctx(
        disposables=(_FakeDisposable(),)
    )

    thread = AgentThread.of(uuid4())

    async with ctx.scope("test"):
        await memory.recall(thread=thread, input=ModelInput.of(MultimodalContent.of("hi")))

    assert events == ["enter", "recall:disposed", "exit"]


def test_agent_memory_with_ctx_without_arguments_returns_self() -> None:
    memory = AgentMemory.volatile()

    assert memory.with_ctx() is memory


@pytest.mark.asyncio
async def test_agent_memory_volatile_accumulates_context_across_turns() -> None:
    memory = AgentMemory.volatile()
    thread = AgentThread.of(uuid4())

    first_input = ModelInput.of(MultimodalContent.of("first"))
    first_recalled = await memory.recall(thread=thread, input=first_input)
    assert first_recalled == (first_input,)

    first_output = ModelOutput.of(MultimodalContent.of("second"))
    await memory.remember(thread=thread, context=(*first_recalled, first_output))

    second_input = ModelInput.of(MultimodalContent.of("third"))
    second_recalled = await memory.recall(thread=thread, input=second_input)
    assert len(second_recalled) == 3
    assert second_recalled[0].content.to_str() == "first"
    assert second_recalled[1].content.to_str() == "second"
    assert second_recalled[2] is second_input

    second_output = ModelOutput.of(MultimodalContent.of("fourth"))
    await memory.remember(thread=thread, context=(*second_recalled, second_output))

    third_recalled = await memory.recall(
        thread=thread,
        input=ModelInput.of(MultimodalContent.of("fifth")),
    )
    assert [element.content.to_str() for element in third_recalled] == [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
    ]


@pytest.mark.asyncio
async def test_agent_memory_volatile_stores_remembered_context_as_snapshot() -> None:
    memory = AgentMemory.volatile()
    thread = AgentThread.of(uuid4())

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
    assert await memory.recall(thread=thread, input=incoming) == (summary_input, incoming)


@pytest.mark.asyncio
async def test_agent_memory_volatile_evicts_least_recently_used_threads() -> None:
    memory = AgentMemory.volatile(threads_limit=2)
    first_thread = AgentThread.of(uuid4())
    second_thread = AgentThread.of(uuid4())
    third_thread = AgentThread.of(uuid4())

    for thread in (first_thread, second_thread, third_thread):
        await memory.remember(
            thread=thread,
            context=(ModelInput.of(MultimodalContent.of("stored")),),
        )

    # first thread exceeded the limit and was evicted
    first_input = ModelInput.of(MultimodalContent.of("fresh"))
    assert await memory.recall(thread=first_thread, input=first_input) == (first_input,)

    # recalling the second thread marks it as recently used...
    second_input = ModelInput.of(MultimodalContent.of("next"))
    assert len(await memory.recall(thread=second_thread, input=second_input)) == 2

    # ...so remembering a new thread evicts the third one instead
    await memory.remember(
        thread=AgentThread.of(uuid4()),
        context=(ModelInput.of(MultimodalContent.of("stored")),),
    )
    third_input = ModelInput.of(MultimodalContent.of("fresh"))
    assert await memory.recall(thread=third_thread, input=third_input) == (third_input,)
    assert len(await memory.recall(thread=second_thread, input=second_input)) == 2
