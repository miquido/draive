from collections.abc import AsyncIterable
from uuid import uuid4

import pytest
from haiway import Meta, ctx

from draive import Agent, AgentIdentity, AgentMessage
from draive.agents import AgentException, AgentsGroup, AgentUnavailable
from draive.agents.types import AgentContext
from draive.models import (
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import MultimodalContent, MultimodalContentPart, TextContent
from draive.steps import Step, StepState
from draive.tools import Toolbox
from draive.utils import ProcessingEvent


def _multimodal_text_of(*parts: MultimodalContentPart) -> str:
    return MultimodalContent.of(*parts).to_str()


@pytest.mark.asyncio
async def test_agent_noop_emits_no_chunks() -> None:
    agent = Agent.noop(
        name="helper",
        description="No operation agent",
    )

    async with ctx.scope("test.agent.noop"):
        chunks = [chunk async for chunk in agent.call(input="hello")]

    assert chunks == []


@pytest.mark.asyncio
async def test_agent_steps_filters_reasoning_chunks() -> None:
    async def execute(
        state: StepState,
    ) -> AsyncIterable[object]:
        yield ProcessingEvent.of("progress", "starting")
        yield ModelReasoningChunk.of(TextContent.of("internal"))
        yield TextContent.of("visible")
        yield state

    agent = Agent.steps(
        Step(execute),
        name="helper",
        description="Test helper",
    )

    async with ctx.scope("test.agent.steps.reasoning"):
        chunks = [chunk async for chunk in agent.call(input="hello")]

    assert len(chunks) == 2
    assert isinstance(chunks[0], ProcessingEvent)
    assert chunks[0].event == "progress"
    assert chunks[0].content.to_str() == "starting"
    assert _multimodal_text_of(chunks[1]) == "visible"


@pytest.mark.asyncio
async def test_agent_call_reuses_context_thread_and_meta() -> None:
    thread = uuid4()
    captured: dict[str, object] = {}

    async def execute(
        message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        context = ctx.state(AgentContext)
        captured["message_thread"] = message.thread
        captured["message_meta"] = message.meta
        captured["context_thread"] = context.thread
        captured["context_meta"] = context.meta
        yield TextContent.of(message.content.to_str())

    agent = Agent(
        identity=AgentIdentity.of(name="helper"),
        executing=execute,
    )
    meta = Meta.of({"source": "outer"})

    async with ctx.scope("test.agent.call", AgentContext.of(thread=thread, meta=meta)):
        chunks = [chunk async for chunk in agent.call(input="hello")]

    assert _multimodal_text_of(*chunks) == "hello"
    assert captured["message_thread"] == thread
    assert captured["context_thread"] == thread
    assert captured["message_meta"] == meta
    assert captured["context_meta"] == meta


@pytest.mark.asyncio
async def test_agent_call_overrides_context_thread_and_merges_meta() -> None:
    outer_thread = uuid4()
    inner_thread = uuid4()
    captured: dict[str, object] = {}

    async def execute(
        message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        context = ctx.state(AgentContext)
        captured["message_thread"] = message.thread
        captured["message_meta"] = message.meta
        captured["context_thread"] = context.thread
        captured["context_meta"] = context.meta
        yield TextContent.of(message.content.to_str())

    agent = Agent(
        identity=AgentIdentity.of(name="helper"),
        executing=execute,
    )

    async with ctx.scope(
        "test.agent.call.override",
        AgentContext.of(thread=outer_thread, meta={"source": "outer", "scope": "root"}),
    ):
        chunks = [
            chunk
            async for chunk in agent.call(
                thread=inner_thread,
                input="hello",
                meta={"source": "inner", "request": "call"},
            )
        ]

    message_meta = captured["message_meta"]
    context_meta = captured["context_meta"]

    assert _multimodal_text_of(*chunks) == "hello"
    assert captured["message_thread"] == inner_thread
    assert captured["context_thread"] == inner_thread
    assert isinstance(message_meta, Meta)
    assert isinstance(context_meta, Meta)
    assert message_meta.get_str("source") == "inner"
    assert message_meta.get_str("scope") == "root"
    assert message_meta.get_str("request") == "call"
    assert context_meta.get_str("source") == "inner"
    assert context_meta.get_str("scope") == "root"
    assert context_meta.get_str("request") == "call"


@pytest.mark.asyncio
async def test_agents_group_request_tool_returns_response_tool_output() -> None:
    agent = Agent.steps(
        Step.emitting(ProcessingEvent.of("progress", "routing"), TextContent.of("done")),
        name="worker",
        description="Worker agent",
    )
    tool = AgentsGroup.of(agent).request_tool()

    async with ctx.scope("test.agent.group.request"):
        chunks = [
            chunk
            async for chunk in Toolbox.of(tool).handle(
                ModelToolRequest.of(
                    "r1",
                    tool="agent_request",
                    arguments={"agent": agent.identity.name, "message": "perform task"},
                )
            )
        ]

    assert len(chunks) == 2
    assert isinstance(chunks[0], ProcessingEvent)
    assert chunks[0].event == "progress"
    assert chunks[0].content.to_str() == "routing"
    assert isinstance(chunks[1], ModelToolResponse)
    assert chunks[1].handling == "response"
    assert chunks[1].status == "success"
    assert chunks[1].result.to_str() == "done"


@pytest.mark.asyncio
async def test_agents_group_handover_tool_streams_direct_output() -> None:
    agent = Agent.steps(
        Step.emitting(ProcessingEvent.of("progress", "routing"), TextContent.of("done")),
        name="worker",
        description="Worker agent",
    )
    tool = AgentsGroup.of(agent).handover_tool()

    async with ctx.scope("test.agent.group.handover"):
        chunks = [
            chunk
            async for chunk in Toolbox.of(tool).handle(
                ModelToolRequest.of(
                    "r1",
                    tool="agent_handover",
                    arguments={"agent": agent.identity.name, "message": "perform task"},
                )
            )
        ]

    assert len(chunks) == 3
    assert isinstance(chunks[0], ProcessingEvent)
    assert chunks[0].event == "progress"
    assert _multimodal_text_of(chunks[1]) == "done"
    assert isinstance(chunks[2], ModelToolResponse)
    assert chunks[2].handling == "output"
    assert chunks[2].status == "success"
    assert chunks[2].result.to_str() == "done"


@pytest.mark.asyncio
async def test_agents_group_tools_raise_for_missing_agent() -> None:
    tool = AgentsGroup.of().request_tool()

    async with ctx.scope("test.agent.group.missing"):
        with pytest.raises(AgentUnavailable, match="Agent `missing` is not defined"):
            _ = [chunk async for chunk in tool.call(agent="missing", message="hello")]


def test_agents_group_rejects_duplicate_agent_names() -> None:
    agent_a = Agent.steps(
        Step.emitting(TextContent.of("a")),
        name="worker",
    )
    agent_b = Agent.steps(
        Step.emitting(TextContent.of("b")),
        name="worker",
    )

    with pytest.raises(ValueError, match="Agent `worker` is already defined"):
        _ = AgentsGroup.of(agent_a, agent_b)


def test_agents_group_rejects_duplicate_agent_uris() -> None:
    shared_uri = "agent://worker"

    async def execute_a(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("a")

    async def execute_b(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("b")

    agent_a = Agent(
        identity=AgentIdentity.of(
            uri=shared_uri,
            name="worker-a",
        ),
        executing=execute_a,
    )
    agent_b = Agent(
        identity=AgentIdentity.of(
            uri=shared_uri,
            name="worker-b",
        ),
        executing=execute_b,
    )

    with pytest.raises(ValueError, match="Agent `agent://worker` is already defined"):
        _ = AgentsGroup.of(agent_a, agent_b)


@pytest.mark.asyncio
async def test_agents_group_call_dispatches_to_selected_agent() -> None:
    thread = uuid4()
    group = AgentsGroup.of(
        Agent.steps(
            Step.emitting(TextContent.of("done")),
            name="worker",
        )
    )

    async with ctx.scope("test.agent.group.call"):
        chunks = [
            chunk
            async for chunk in group.call(
                "worker",
                thread=thread,
                input="perform task",
                meta={"source": "caller"},
            )
        ]

    assert len(chunks) == 1
    assert _multimodal_text_of(*chunks) == "done"


@pytest.mark.asyncio
async def test_agents_group_call_raises_for_missing_agent() -> None:
    group = AgentsGroup.of()

    async with ctx.scope("test.agent.group.call.missing"):
        with pytest.raises(AgentUnavailable, match="Agent `missing` is not defined"):
            _ = [chunk async for chunk in group.call("missing", input="hello")]


@pytest.mark.asyncio
async def test_agents_group_call_raises_for_unbound_placeholder_agent() -> None:
    identity = AgentIdentity.of(
        name="worker",
        description="Deferred worker",
    )
    group = AgentsGroup.of(identity)

    async with ctx.scope("test.agent.group.call.unbound"):
        with pytest.raises(AgentUnavailable, match="Agent `worker` is not defined"):
            _ = [chunk async for chunk in group.call("worker", input="hello")]


@pytest.mark.asyncio
async def test_agents_group_bind_replaces_placeholder_agent_with_same_identity() -> None:
    identity = AgentIdentity.of(
        name="worker",
        description="Deferred worker",
    )
    group = AgentsGroup.of(identity)

    async def execute(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("bound")

    group.bind(
        Agent(
            identity=identity,
            executing=execute,
        )
    )

    async with ctx.scope("test.agent.group.bind"):
        chunks = [chunk async for chunk in group.call("worker", input="perform task")]

    assert len(chunks) == 1
    assert _multimodal_text_of(*chunks) == "bound"


def test_agents_group_bind_rejects_binding_different_identity_with_same_name() -> None:
    group = AgentsGroup.of(
        AgentIdentity.of(
            name="worker",
            description="Deferred worker",
        )
    )

    with pytest.raises(AgentException, match="AgentGroup agents can't be extended"):
        group.bind(
            Agent.steps(
                Step.emitting(TextContent.of("bound")),
                name="worker",
                description="Deferred worker",
            )
        )


def test_agents_group_bind_rejects_extending_with_new_name() -> None:
    group = AgentsGroup.of()

    with pytest.raises(AgentException, match="AgentGroup agents can't be extended"):
        group.bind(
            Agent.steps(
                Step.emitting(TextContent.of("new")),
                name="worker",
            )
        )


def test_agents_group_bind_rejects_rebinding_existing_agent() -> None:
    async def execute_first(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("first")

    async def execute_second(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("second")

    identity = AgentIdentity.of(name="worker")
    group = AgentsGroup.of(
        Agent(
            identity=identity,
            executing=execute_first,
        )
    )

    with pytest.raises(AgentException, match="AgentGroup agents can't be redefined"):
        group.bind(  # same declared identity, but already bound
            Agent(
                identity=identity,
                executing=execute_second,
            )
        )
