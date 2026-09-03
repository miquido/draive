from collections.abc import AsyncIterable, MutableSequence, Sequence
from typing import Any

import pytest
from haiway import ctx

from draive.agents import Agent, AgentMemory
from draive.conversation import Conversation, ConversationEvent
from draive.conversation.state import ConversationMemory
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
from draive.multimodal import MultimodalContent, TextContent
from draive.steps import Step
from draive.tools import Toolbox, tool


class _ScriptedModel:
    """Model emitting predefined chunks per completion, recording each request."""

    def __init__(
        self,
        *turns: Sequence[ModelOutputChunk],
    ) -> None:
        self.turns: Sequence[Sequence[ModelOutputChunk]] = turns
        self.contexts: MutableSequence[ModelContext] = []
        self.tools: MutableSequence[ModelTools] = []

    def state(self) -> GenerativeModel:
        def generating(
            *,
            instructions: str,
            tools: ModelTools,
            context: Sequence[ModelContextElement],
            output: Any = "auto",
            **extra: Any,
        ) -> AsyncIterable[ModelOutputChunk]:
            _ = (instructions, output, extra)
            self.contexts.append(tuple(context))
            self.tools.append(tools)
            chunks = self.turns[min(len(self.contexts) - 1, len(self.turns) - 1)]

            async def stream() -> AsyncIterable[ModelOutputChunk]:
                for chunk in chunks:
                    yield chunk

            return stream()

        return GenerativeModel(generating=generating)


def _request(
    identifier: str,
    tool_name: str,
    /,
    **arguments: Any,
) -> ModelToolRequest:
    return ModelToolRequest.of(identifier, tool=tool_name, arguments=arguments)


def _dependent_tools(calls: MutableSequence[str]) -> Sequence[Any]:
    @tool(description="Resolve a user id by name")
    async def lookup_user(name: str) -> str:
        calls.append(f"lookup_user({name})")
        return "u-1"

    @tool(description="Fetch orders for a user id")
    async def fetch_orders(user_id: str) -> str:
        calls.append(f"fetch_orders({user_id})")
        return "2 orders"

    return (lookup_user, fetch_orders)


def _two_turn_script() -> _ScriptedModel:
    return _ScriptedModel(
        (_request("call-1", "lookup_user", name="ada"),),
        (_request("call-2", "fetch_orders", user_id="u-1"),),
        (TextContent.of("Ada has 2 orders."),),
    )


@pytest.mark.asyncio
async def test_conversation_runs_dependent_tools_across_turns() -> None:
    calls: MutableSequence[str] = []
    model = _two_turn_script()

    async with ctx.scope("test", model.state()):
        chunks = [
            chunk
            async for chunk in Conversation.completion(
                message="orders?",
                tools=Toolbox.of(_dependent_tools(calls)),
                memory=ConversationMemory.disabled,
            )
        ]

    assert list(calls) == ["lookup_user(ada)", "fetch_orders(u-1)"]
    # every completion receives the context accumulated by preceding turns
    assert [len(context) for context in model.contexts] == [1, 3, 5]

    events = [chunk for chunk in chunks if isinstance(chunk, ConversationEvent)]
    requested = [
        event.content.artifact["tool"]
        for event in events
        if event.event == "tool_request" and event.content is not None
    ]
    assert requested == ["lookup_user", "fetch_orders"]
    assert sum(1 for event in events if event.event == "tool_response") == 2
    assert (
        MultimodalContent.of(
            *[chunk for chunk in chunks if isinstance(chunk, TextContent)]
        ).to_str()
        == "Ada has 2 orders."
    )


@pytest.mark.asyncio
async def test_step_loop_accumulates_tool_exchanges_in_context() -> None:
    calls: MutableSequence[str] = []
    model = _two_turn_script()

    async with ctx.scope("test", model.state()):
        state = await Step.sequence(
            Step.appending_context(ModelInput.of(MultimodalContent.of("orders?"))),
            Step.looping_completion(tools=Toolbox.of(_dependent_tools(calls))),
        ).process()

    assert list(calls) == ["lookup_user(ada)", "fetch_orders(u-1)"]
    assert [type(element).__name__ for element in state.context] == [
        "ModelInput",
        "ModelOutput",
        "ModelInput",
        "ModelOutput",
        "ModelInput",
        "ModelOutput",
    ]
    assert [
        request.tool
        for element in state.context
        if isinstance(element, ModelOutput)
        for request in element.tool_requests
    ] == ["lookup_user", "fetch_orders"]


@pytest.mark.asyncio
async def test_agent_remembers_whole_multi_turn_tool_exchange() -> None:
    calls: MutableSequence[str] = []
    remembered: MutableSequence[ModelContext] = []
    model = _two_turn_script()

    async def recalling(thread: Any, context: ModelContext, **extra: Any) -> ModelContext:
        _ = (thread, extra)
        return context

    async def remembering(thread: Any, context: ModelContext, **extra: Any) -> None:
        _ = (thread, extra)
        remembered.append(tuple(context))

    agent = Agent.generative(
        agent="orders",
        instructions="use tools",
        tools=Toolbox.of(_dependent_tools(calls)),
        memory=AgentMemory(recalling=recalling, remembering=remembering),
    )

    async with ctx.scope("test", model.state()):
        chunks = [chunk async for chunk in agent.call(input="orders?")]

    assert list(calls) == ["lookup_user(ada)", "fetch_orders(u-1)"]
    assert len(remembered) == 1
    assert [type(element).__name__ for element in remembered[0]] == [
        "ModelInput",
        "ModelOutput",
        "ModelInput",
        "ModelOutput",
        "ModelInput",
        "ModelOutput",
    ]
    assert (
        MultimodalContent.of(
            *[chunk for chunk in chunks if isinstance(chunk, TextContent)]
        ).to_str()
        == "Ada has 2 orders."
    )


@pytest.mark.asyncio
async def test_agent_does_not_remember_abandoned_turn() -> None:
    remembered: MutableSequence[int] = []
    model = _ScriptedModel((TextContent.of("first"), TextContent.of("second")))

    async def recalling(thread: Any, context: ModelContext, **extra: Any) -> ModelContext:
        _ = (thread, extra)
        return context

    async def remembering(thread: Any, context: ModelContext, **extra: Any) -> None:
        _ = (thread, extra)
        remembered.append(len(tuple(context)))

    agent = Agent.generative(
        agent="abandoning",
        instructions="",
        memory=AgentMemory(recalling=recalling, remembering=remembering),
    )

    async with ctx.scope("test", model.state()):
        stream = agent.call(input="hello")
        async for _ in stream:
            break  # abandon before the turn completes

        await stream.aclose()
        assert not remembered

        drained = [chunk async for chunk in agent.call(input="hello again")]

    assert drained
    assert remembered == [2]


@pytest.mark.asyncio
async def test_parallel_tool_requests_produce_single_input_element() -> None:
    calls: MutableSequence[str] = []
    model = _ScriptedModel(
        (
            _request("call-1", "lookup_user", name="ada"),
            _request("call-2", "fetch_orders", user_id="u-9"),
        ),
        (TextContent.of("done"),),
    )

    async with ctx.scope("test", model.state()):
        state = await Step.sequence(
            Step.appending_context(ModelInput.of(MultimodalContent.of("both"))),
            Step.looping_completion(tools=Toolbox.of(_dependent_tools(calls))),
        ).process()

    assert sorted(calls) == ["fetch_orders(u-9)", "lookup_user(ada)"]
    response_inputs = [
        element
        for element in state.context
        if isinstance(element, ModelInput) and element.tool_responses
    ]
    assert len(response_inputs) == 1
    assert len(response_inputs[0].tool_responses) == 2


@pytest.mark.asyncio
async def test_failing_tool_keeps_loop_running_with_error_response() -> None:
    model = _ScriptedModel(
        (_request("call-1", "broken"),),
        (TextContent.of("recovered"),),
    )

    @tool(description="Always fails")
    async def broken() -> str:
        raise RuntimeError("tool exploded")

    async with ctx.scope("test", model.state()):
        state = await Step.sequence(
            Step.appending_context(ModelInput.of(MultimodalContent.of("try"))),
            Step.looping_completion(tools=Toolbox.of([broken])),
        ).process()

    responses = [
        response
        for element in state.context
        if isinstance(element, ModelInput)
        for response in element.tool_responses
    ]
    assert len(responses) == 1
    assert responses[0].status == "error"
    assert len(model.contexts) == 2  # the loop continued after the failure


@pytest.mark.asyncio
async def test_direct_output_tool_ends_the_loop() -> None:
    model = _ScriptedModel(
        (_request("call-1", "final_answer", value="direct"),),
        (TextContent.of("unreachable"),),
    )

    @tool(description="Provide the final answer", handling="output")
    async def final_answer(value: str) -> str:
        return value

    async with ctx.scope("test", model.state()):
        content = await Step.sequence(
            Step.appending_context(ModelInput.of(MultimodalContent.of("answer"))),
            Step.looping_completion(tools=Toolbox.of([final_answer])),
        ).run()

    assert "direct" in content.to_str()
    assert len(model.contexts) == 1  # no further completion after direct output


@pytest.mark.asyncio
async def test_tool_suggestion_applies_to_first_iteration_only() -> None:
    calls: MutableSequence[str] = []
    model = _two_turn_script()
    tools = _dependent_tools(calls)

    async with ctx.scope("test", model.state()):
        await Step.sequence(
            Step.appending_context(ModelInput.of(MultimodalContent.of("go"))),
            Step.looping_completion(tools=Toolbox.of(tools, suggesting=tools[0])),
        ).process()

    selections = [
        item.selection if isinstance(item.selection, str) else item.selection.name
        for item in model.tools
    ]
    assert selections == ["lookup_user", "auto", "auto"]
