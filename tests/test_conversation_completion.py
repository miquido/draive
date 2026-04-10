from collections.abc import AsyncIterable, Sequence
from typing import Any

import pytest
from haiway import ctx

from draive.conversation import Conversation
from draive.conversation.state import ConversationMemory
from draive.conversation.types import ConversationAssistantTurn, ConversationUserTurn
from draive.models import (
    GenerativeModel,
    ModelContextElement,
    ModelInput,
    ModelOutput,
    ModelOutputChunk,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.tools import tool


async def _single_text_chunk(text: str) -> AsyncIterable[ModelOutputChunk]:
    yield TextContent.of(text)


async def _stream_of(*chunks: ModelOutputChunk) -> AsyncIterable[ModelOutputChunk]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_conversation_completion_emits_stream_chunks() -> None:
    def mock_generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: str | type | Sequence[str],
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        _ = (instructions, tools, context, output, extra)
        return _single_text_chunk("Hello from assistant")

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
        stream = Conversation.completion(
            message="Hi",
            memory=ConversationMemory.disabled,
        )
        chunks = [chunk async for chunk in stream]

    assert MultimodalContent.of(*chunks).to_str() == "Hello from assistant"


@pytest.mark.asyncio
async def test_conversation_completion_loops_tool_response_into_followup_completion() -> None:
    calls = 0

    @tool
    async def echo(value: str) -> str:
        return f"TOOL:{value}"

    def mock_generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: str | type | Sequence[str],
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal calls
        _ = (instructions, tools, output, extra)
        calls += 1
        if calls == 1:
            assert len(context) == 1
            assert isinstance(context[0], ModelInput)
            assert context[0].content.to_str() == "Hi"
            return _stream_of(
                TextContent.of("Checking"),
                ModelToolRequest.of("call-1", tool="echo", arguments={"value": "x"}),
            )

        assert len(context) == 3
        assert isinstance(context[1], ModelOutput)
        assert context[1].tool_requests[0].identifier == "call-1"
        assert isinstance(context[2], ModelInput)
        assert context[2].tool_responses[0].identifier == "call-1"
        assert context[2].tool_responses[0].content.to_str() == "TOOL:x"
        return _stream_of(TextContent.of("Final answer"))

    memory = ConversationMemory.volatile()
    async with ctx.scope(
        "test",
        GenerativeModel(generating=mock_generating),
        Conversation(),
    ):
        stream = Conversation.completion(
            message="Hi",
            tools=[echo],
            memory=memory,
        )
        chunks = [chunk async for chunk in stream]

    assert calls == 2
    assert chunks[0].text == "Checking"
    assert chunks[1].event == "tool_request"
    assert chunks[2].event == "tool_response"
    assert chunks[3].text == "Final answer"

    remembered_turns = await memory.fetch()
    assert len(remembered_turns) == 2
    assistant_turn = remembered_turns[1]
    assert isinstance(assistant_turn, ConversationAssistantTurn)
    assert assistant_turn.content[0].to_str() == "Checking"
    assert assistant_turn.content[1].event == "tool_request"
    assert assistant_turn.content[2].event == "tool_response"
    assert assistant_turn.content[3].to_str() == "Final answer"


@pytest.mark.asyncio
async def test_conversation_completion_output_tool_stops_loop_and_persists_output() -> None:
    calls = 0

    @tool(handling="output")
    async def amplify(value: str) -> str:
        return f"OUT:{value}"

    def mock_generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: str | type | Sequence[str],
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal calls
        _ = (instructions, tools, context, output, extra)
        calls += 1
        return _stream_of(ModelToolRequest.of("call-1", tool="amplify", arguments={"value": "x"}))

    memory = ConversationMemory.volatile()
    async with ctx.scope(
        "test",
        GenerativeModel(generating=mock_generating),
        Conversation(),
    ):
        stream = Conversation.completion(
            message="Hi",
            tools=[amplify],
            memory=memory,
        )
        chunks = [chunk async for chunk in stream]

    assert calls == 1
    assert chunks[0].event == "tool_request"
    assert chunks[1].text == "OUT:x"
    assert chunks[2].event == "tool_response"

    context = await memory.recall()
    assert len(context) == 4
    assert isinstance(context[0], ModelInput)
    assert isinstance(context[1], ModelOutput)
    assert context[1].tool_requests[0].identifier == "call-1"
    assert isinstance(context[2], ModelInput)
    assert context[2].tool_responses[0].identifier == "call-1"
    assert isinstance(context[3], ModelOutput)
    assert context[3].content.to_str() == "OUT:x"


@pytest.mark.asyncio
async def test_conversation_completion_reuses_memory_with_previous_tool_turn() -> None:
    @tool
    async def echo(value: str) -> str:
        return f"TOOL:{value}"

    memory = ConversationMemory.volatile()
    first_calls = 0

    def first_generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: str | type | Sequence[str],
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal first_calls
        _ = (instructions, tools, output, extra)
        first_calls += 1
        if first_calls == 1:
            return _stream_of(ModelToolRequest.of("call-1", tool="echo", arguments={"value": "x"}))

        assert len(context) == 3
        assert isinstance(context[2], ModelInput)
        assert context[2].tool_responses[0].content.to_str() == "TOOL:x"
        return _stream_of(TextContent.of("First final"))

    async with ctx.scope(
        "test.first",
        GenerativeModel(generating=first_generating),
        Conversation(),
    ):
        first_stream = Conversation.completion(
            message="Hi",
            tools=[echo],
            memory=memory,
        )
        _ = [chunk async for chunk in first_stream]

    second_calls = 0

    def second_generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: str | type | Sequence[str],
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal second_calls
        _ = (instructions, tools, output, extra)
        second_calls += 1
        assert len(context) == 5
        assert isinstance(context[0], ModelInput)
        assert context[0].content.to_str() == "Hi"
        assert isinstance(context[1], ModelOutput)
        assert context[1].tool_requests[0].identifier == "call-1"
        assert isinstance(context[2], ModelInput)
        assert context[2].tool_responses[0].identifier == "call-1"
        assert isinstance(context[3], ModelOutput)
        assert context[3].content.to_str() == "First final"
        assert isinstance(context[4], ModelInput)
        assert context[4].content.to_str() == "Follow-up"
        return _stream_of(TextContent.of("Second answer"))

    async with ctx.scope(
        "test.second",
        GenerativeModel(generating=second_generating),
        Conversation(),
    ):
        second_stream = Conversation.completion(
            message="Follow-up",
            tools=[echo],
            memory=memory,
        )
        chunks = [chunk async for chunk in second_stream]

    assert second_calls == 1
    assert MultimodalContent.of(*chunks).to_str() == "Second answer"


@pytest.mark.asyncio
async def test_conversation_completion_links_new_turns_in_memory_order() -> None:
    def mock_generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: str | type | Sequence[str],
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        _ = (instructions, tools, context, output, extra)
        return _single_text_chunk("linked answer")

    memory = ConversationMemory.volatile()
    existing_turn = ConversationUserTurn.of(MultimodalContent.of("earlier"))
    await memory.remember(existing_turn)

    async with ctx.scope(
        "test",
        GenerativeModel(generating=mock_generating),
        Conversation(),
    ):
        stream = Conversation.completion(
            message="Hi",
            memory=memory,
        )
        _ = [chunk async for chunk in stream]

    remembered_turns = await memory.fetch()
    assert len(remembered_turns) == 3
    user_turn = remembered_turns[1]
    assistant_turn = remembered_turns[2]
    assert remembered_turns[0].identifier == existing_turn.identifier
    assert user_turn.turn == "user"
    assert assistant_turn.turn == "assistant"
