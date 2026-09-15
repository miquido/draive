from collections.abc import AsyncGenerator, AsyncIterable, Collection, Sequence
from typing import Any
from uuid import uuid4

import pytest
from haiway import Meta, ctx

from draive import Agent, AgentIdentity, AgentMessage
from draive.agents import AgentException, AgentMemory, AgentsGroup, AgentThread, AgentUnavailable
from draive.models import (
    GenerativeModel,
    ModelContextElement,
    ModelOutput,
    ModelOutputChunk,
    ModelReasoningChunk,
    ModelToolHandling,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import MultimodalContent, MultimodalContentPart, TextContent
from draive.skills import Skill
from draive.steps import Step, StepState
from draive.tools import Toolbox
from draive.utils import ProcessingEvent


def _multimodal_text_of(*parts: MultimodalContentPart) -> str:
    return MultimodalContent.of(*parts).to_str()


@pytest.mark.asyncio
async def test_agent_noop_emits_no_chunks() -> None:
    agent = Agent.noop(
        agent="helper",
        description="No operation agent",
    )

    async with ctx.scope("test.agent.noop"):
        chunks = [chunk async for chunk in agent.call(input="hello")]

    assert chunks == []


@pytest.mark.parametrize("handling, suffix", [("response", "request"), ("output", "handover")])
def test_agent_tool_normalizes_generated_name(handling: ModelToolHandling, suffix: str) -> None:
    first = Agent.noop("friendly helper/with a very long name " * 3).as_tool(
        handling=handling,
    )
    second = Agent.noop("friendly helper with a very long name " * 3).as_tool(
        handling=handling,
    )

    normalized_name = "friendly_helper_with_a_very_long_name_" * 3
    assert first.name == second.name == f"agent_{normalized_name.rstrip('_')}_{suffix}"


def test_agent_tool_preserves_valid_generated_name() -> None:
    assert Agent.noop("friendly-helper_2").as_tool().name == "agent_friendly-helper_2_request"


@pytest.mark.asyncio
async def test_generative_agent_groups_interleaved_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = (
        ModelToolRequest.of("r1", tool="shared", arguments={}),
        ModelToolRequest.of("r2", tool="shared", arguments={}),
    )

    async def handle(
        self: Toolbox,
        pending: Collection[ModelToolRequest],
    ) -> AsyncGenerator[TextContent | ModelToolResponse]:
        assert tuple(pending) == requests
        yield TextContent.of("a1", meta={"request": "r2"})
        yield TextContent.of("b1", meta={"request": "r1"})
        yield TextContent.of("a2", meta={"request": "r2"})
        yield TextContent.of("b2", meta={"request": "r1"})
        for request in requests:
            yield ModelToolResponse(
                identifier=request.identifier,
                tool=request.tool,
                status="success",
                content=MultimodalContent.empty,
            )

    monkeypatch.setattr(Toolbox, "handle", handle)

    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        for request in requests:
            yield request

    thread = uuid4()
    memory = AgentMemory.volatile()
    agent = Agent.generative(
        "helper",
        instructions="Use tools",
        memory=memory,
    )
    async with ctx.scope("test", GenerativeModel(generating=generating)):
        chunks = [chunk async for chunk in agent.call(thread=thread, input="hello")]

    assert MultimodalContent.of(*chunks).to_str() == "a1b1a2b2"
    context = await memory.recall(
        thread=AgentThread.of(thread, agent_uri=agent.identity.uri),
        context=(),
    )
    output = context[-1]
    assert isinstance(output, ModelOutput)
    assert output.content.to_str() == "a1a2b1b2"


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
        agent="helper",
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
        context = ctx.state(AgentThread)
        captured["message_thread"] = message.thread
        captured["message_meta"] = message.meta
        captured["context_thread"] = context.identifier
        captured["context_meta"] = context.meta
        yield TextContent.of(message.content.to_str())

    agent = Agent(
        identity=AgentIdentity.of(name="helper"),
        executing=execute,
    )
    meta = Meta.of({"source": "outer"})

    async with ctx.scope(
        "test.agent.call",
        AgentThread.of(identifier=thread, agent_uri="agent://outer", meta=meta),
    ):
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
        context = ctx.state(AgentThread)
        captured["message_thread"] = message.thread
        captured["message_meta"] = message.meta
        captured["context_thread"] = context.identifier
        captured["context_meta"] = context.meta
        yield TextContent.of(message.content.to_str())

    agent = Agent(
        identity=AgentIdentity.of(name="helper"),
        executing=execute,
    )

    async with ctx.scope(
        "test.agent.call.override",
        AgentThread.of(
            identifier=outer_thread,
            agent_uri="agent://outer",
            meta={"source": "outer", "scope": "root"},
        ),
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
        agent="worker",
        description="Worker agent",
    )
    tool = AgentsGroup.of(agent).as_tool()

    async with ctx.scope("test.agent.group.request"):
        chunks = [
            chunk
            async for chunk in Toolbox.of(tool).handle(
                (
                    ModelToolRequest.of(
                        "r1",
                        tool="agent_request",
                        arguments={"agent": agent.identity.name, "task": "perform task"},
                    ),
                )
            )
        ]

    assert len(chunks) == 2
    assert isinstance(chunks[0], ProcessingEvent)
    assert chunks[0].event == "progress"
    assert chunks[0].content.to_str() == "routing"
    assert isinstance(chunks[1], ModelToolResponse)
    assert chunks[1].status == "success"
    assert chunks[1].content.to_str() == "done"


@pytest.mark.asyncio
async def test_agents_group_handover_tool_streams_direct_output() -> None:
    agent = Agent.steps(
        Step.emitting(ProcessingEvent.of("progress", "routing"), TextContent.of("done")),
        agent="worker",
        description="Worker agent",
    )
    tool = AgentsGroup.of(agent).as_tool(handling="output")

    async with ctx.scope("test.agent.group.handover"):
        chunks = [
            chunk
            async for chunk in Toolbox.of(tool).handle(
                (
                    ModelToolRequest.of(
                        "r1",
                        tool="agent_handover",
                        arguments={"agent": agent.identity.name, "task": "perform task"},
                    ),
                )
            )
        ]

    assert len(chunks) == 3
    assert isinstance(chunks[0], ProcessingEvent)
    assert chunks[0].event == "progress"
    assert _multimodal_text_of(chunks[1]) == "done"
    assert isinstance(chunks[2], ModelToolResponse)
    assert chunks[2].status == "success"
    assert chunks[2].content.to_str() == "done"


@pytest.mark.asyncio
async def test_agents_group_tools_raise_for_missing_agent() -> None:
    tool = AgentsGroup.of().as_tool()

    async with ctx.scope("test.agent.group.missing"):
        with pytest.raises(AgentUnavailable, match="Agent `missing` is not defined"):
            _ = [chunk async for chunk in tool.call(agent="missing", task="hello")]


def test_agents_group_rejects_duplicate_agent_names() -> None:
    agent_a = Agent.steps(
        Step.emitting(TextContent.of("a")),
        agent="worker",
    )
    agent_b = Agent.steps(
        Step.emitting(TextContent.of("b")),
        agent="worker",
    )

    with pytest.raises(AssertionError):
        _ = AgentsGroup.of(agent_a, agent_b)


def test_agents_group_rejects_duplicate_placeholder_names() -> None:
    with pytest.raises(AssertionError):
        _ = AgentsGroup.of(
            AgentIdentity.of(uri="agent://worker-a", name="worker"),
            AgentIdentity.of(uri="agent://worker-b", name="worker"),
        )


def test_agents_group_rejects_duplicate_placeholder_uris() -> None:
    with pytest.raises(ValueError, match="Agent `agent://worker` is already defined"):
        _ = AgentsGroup.of(
            AgentIdentity.of(uri="agent://worker", name="worker-a"),
            AgentIdentity.of(uri="agent://worker", name="worker-b"),
        )


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


def test_agent_skill_constructor_uses_skill_identity_and_meta() -> None:
    skill = Skill.of(
        "code-review",
        description="Review source code for issues and regressions.",
        instructions="Analyze code changes and report findings.",
        meta={"source": "skill", "scope": "skill"},
    )

    agent = Agent.from_skill(
        skill,
        meta={"source": "override", "request": "constructor"},
    )

    assert agent.identity.name == "code-review"
    assert agent.identity.description == "Review source code for issues and regressions."
    assert agent.identity.meta.get_str("source") == "override"
    assert agent.identity.meta.get_str("scope") == "skill"
    assert agent.identity.meta.get_str("request") == "constructor"


@pytest.mark.asyncio
async def test_agents_group_call_dispatches_to_selected_agent() -> None:
    thread = uuid4()
    group = AgentsGroup.of(
        Agent.steps(
            Step.emitting(TextContent.of("done")),
            agent="worker",
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
async def test_agents_group_snapshots_available_mapping() -> None:
    available: dict[str, Agent] = {}
    group = AgentsGroup(available)
    available["worker"] = Agent.noop("worker")

    async with ctx.scope("test.agent.group.mapping"):
        with pytest.raises(AgentUnavailable, match="Agent `worker` is not defined"):
            _ = [chunk async for chunk in group.call("worker", input="hello")]


@pytest.mark.asyncio
async def test_agents_group_call_raises_for_unbound_placeholder_agent() -> None:
    identity = AgentIdentity.of(
        name="worker",
        description="Deferred worker",
    )
    group = AgentsGroup.of(identity)

    async with ctx.scope("test.agent.group.call.unbound"):
        with pytest.raises(AgentUnavailable, match="Agent execution method undefined!"):
            _ = [chunk async for chunk in group.call("worker", input="hello")]


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["worker", "agent://worker"])
async def test_agents_group_bind_replaces_placeholder_agent_with_same_identity(
    reference: str,
) -> None:
    identity = AgentIdentity.of(
        uri="agent://worker",
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
        chunks = [chunk async for chunk in group.call(reference, input="perform task")]

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
                agent="worker",
                description="Deferred worker",
            )
        )


def test_agents_group_bind_rejects_extending_with_new_name() -> None:
    group = AgentsGroup.of()

    with pytest.raises(AgentException, match="AgentGroup agents can't be extended"):
        group.bind(
            Agent.steps(
                Step.emitting(TextContent.of("new")),
                agent="worker",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["worker", "agent://worker"])
async def test_agents_group_bind_replaces_existing_agent(reference: str) -> None:
    async def execute_first(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("first")

    async def execute_second(
        _message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        yield TextContent.of("second")

    identity = AgentIdentity.of(uri="agent://worker", name="worker")
    group = AgentsGroup.of(
        Agent(
            identity=identity,
            executing=execute_first,
        )
    )

    tool = group.as_tool()
    group.bind(
        Agent(
            identity=identity,
            executing=execute_second,
        )
    )

    async with ctx.scope("test.agent.group.rebind"):
        chunks = [chunk async for chunk in group.call(reference, input="hello")]
        tool_chunks = [chunk async for chunk in tool.call(agent="worker", task="hello")]

    assert _multimodal_text_of(*chunks) == "second"
    assert tool_chunks == [TextContent.of("second")]


@pytest.mark.parametrize("handling", ["response", "output"])
def test_agents_group_tool_lists_each_agent_once(handling: ModelToolHandling) -> None:
    group = AgentsGroup.of(AgentIdentity.of(name="worker", description="Deferred worker"))
    tool = group.as_tool(handling=handling)

    assert tool.description is not None
    assert tool.description.count('<agent name="worker">') == 1
    parameters = tool.parameters
    assert parameters is not None
    assert parameters["properties"]["agent"] == {
        "type": "string",
        "enum": ("worker",),
        "description": "Selected agent name",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "before, after",
    [("before", "after"), ("agent://worker", "after"), ("before", "agent://worker")],
)
async def test_agents_group_bind_renames_agent_with_same_uri(before: str, after: str) -> None:
    identity = AgentIdentity.of(uri="agent://worker", name=before)
    group = AgentsGroup.of(identity)
    group.bind(
        Agent.steps(
            Step.emitting(TextContent.of("done")),
            agent=AgentIdentity.of(uri=identity.uri, name=after),
        )
    )

    async with ctx.scope("test.agent.group.rename"):
        by_uri = [chunk async for chunk in group.call(identity.uri, input="hello")]
        by_name = [chunk async for chunk in group.call(after, input="hello")]
        if before != identity.uri:
            with pytest.raises(AgentUnavailable, match=f"Agent `{before}` is not defined"):
                _ = [chunk async for chunk in group.call(before, input="hello")]

    assert by_uri == by_name == [TextContent.of("done")]


@pytest.mark.parametrize("name", ["other", "agent://other"])
def test_agents_group_bind_rejects_conflicting_name(name: str) -> None:
    identity = AgentIdentity.of(uri="agent://worker", name="worker")
    group = AgentsGroup.of(identity, AgentIdentity.of(uri="agent://other", name="other"))

    with pytest.raises(AgentException, match="is already defined"):
        group.bind(Agent.noop(AgentIdentity.of(uri=identity.uri, name=name)))
