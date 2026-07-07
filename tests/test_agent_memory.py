from types import TracebackType
from uuid import uuid4

import pytest
from haiway import State, ctx

from draive.agents import AgentMemory, AgentThread
from draive.models import ModelContext, ModelInput
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
