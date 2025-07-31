from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import uuid4

from pytest import mark

from draive import (
    LMM,
    Conversation,
    ConversationEvent,
    ConversationMemory,
    ConversationMessage,
    ConversationMessageChunk,
    LMMCompletion,
    LMMContextElement,
    LMMStreamChunk,
    LMMToolRequest,
    MultimodalContent,
    Tool,
    Toolbox,
    ctx,
    tool,
)
from draive.lmm import LMMToolRequests


class MockLMMTracker:
    """Tracks LMM calls for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_context = None
        self.last_instruction = None
        self.last_tools = None


def extract_text_content(content: MultimodalContent) -> str:
    """Extract text from potentially nested MultimodalContent."""
    return "".join(
        part.text
        if hasattr(part, "text")
        else part.content.to_str()
        if hasattr(part, "content")
        else str(part)
        for part in content.parts
    )


@mark.asyncio
async def test_conversation_completion_simple_response():
    """Test basic conversation completion with simple text response."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        context: Sequence[LMMContextElement],
        tools: Sequence[Tool] | None = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.call_count += 1
        tracker.last_context = context
        tracker.last_instruction = instruction
        tracker.last_tools = tools
        return LMMCompletion.of("Hello! How can I help you?")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            input="Hi there!",
            memory=None,
            tools=None,
        )

        assert isinstance(result, ConversationMessage)
        assert result.role == "model"
        assert extract_text_content(result.content) == "Hello! How can I help you?"
        assert tracker.call_count == 1


@mark.asyncio
async def test_conversation_completion_with_memory():
    """Test conversation completion with memory context."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        context: Sequence[LMMContextElement],
        **extra: Any,
    ) -> LMMCompletion:
        tracker.call_count += 1
        tracker.last_context = context
        return LMMCompletion.of("I can see we talked before.")

    memory_messages = [
        ConversationMessage.user("Previous message"),
        ConversationMessage.model("Previous response"),
    ]

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            input="Do you remember our conversation?",
            memory=memory_messages,
            tools=None,
        )

        assert isinstance(result, ConversationMessage)
        assert extract_text_content(result.content) == "I can see we talked before."
        assert tracker.call_count == 1
        # Check that memory was included in context
        assert len(tracker.last_context or []) == 3  # 2 memory + 1 input


@mark.asyncio
async def test_conversation_completion_with_instruction():
    """Test conversation completion with custom instruction."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_instruction = instruction
        return LMMCompletion.of("I'm a helpful assistant!")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            instruction="You are a helpful assistant",
            input="Who are you?",
            memory=None,
            tools=None,
        )

        assert extract_text_content(result.content) == "I'm a helpful assistant!"
        assert tracker.last_instruction == "You are a helpful assistant"


@mark.asyncio
async def test_conversation_completion_with_tools():
    """Test conversation completion with tool usage."""
    call_count = 0

    @tool
    async def calculate(a: int, b: int) -> int:
        """Calculate sum of two numbers."""
        return a + b

    tool_request = LMMToolRequest(
        identifier=str(uuid4()),
        tool="calculate",
        arguments={"a": 5, "b": 3},
    )

    async def mock_completion(**kwargs) -> LMMCompletion | LMMToolRequests:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call requests tool usage
            return LMMToolRequests.of([tool_request])
        else:
            # After tool response, return final result
            return LMMCompletion.of("The sum is 8")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            input="What is 5 + 3?",
            memory=None,
            tools=[calculate],
        )

        assert extract_text_content(result.content) == "The sum is 8"
        assert call_count == 2  # Initial request + after tool response


@mark.asyncio
async def test_conversation_completion_stream():
    """Test conversation completion with streaming response."""

    async def mock_completion(
        *,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk]:
        if not stream:
            raise ValueError("Expected streaming")

        async def generate_stream():
            yield LMMStreamChunk.of("Hello ")
            yield LMMStreamChunk.of("world!")
            yield LMMStreamChunk.of("", eod=True)

        return generate_stream()

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        chunks = []
        async for element in await Conversation.completion(
            input="Say hello",
            memory=None,
            tools=None,
            stream=True,
        ):
            chunks.append(element)

        assert len(chunks) == 3
        assert all(isinstance(chunk, ConversationMessageChunk) for chunk in chunks)
        assert chunks[0].content.to_str() == "Hello "
        assert chunks[1].content.to_str() == "world!"
        assert chunks[2].eod is True


@mark.asyncio
async def test_conversation_completion_stream_with_tools():
    """Test streaming conversation with tool calls."""
    call_count = 0

    @tool
    async def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    tool_request = LMMToolRequest(
        identifier=str(uuid4()),
        tool="get_weather",
        arguments={"city": "Paris"},
    )

    async def mock_completion(
        *,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk | LMMToolRequest]:
        nonlocal call_count
        call_count += 1

        if not stream:
            raise ValueError("Expected streaming")

        async def generate_stream():
            if call_count == 1:
                # First stream includes tool request
                yield LMMStreamChunk.of("Let me check the weather")
                yield tool_request
                yield LMMStreamChunk.of("", eod=True)
            else:
                # After tool, return final response
                yield LMMStreamChunk.of("The weather in Paris is sunny!")
                yield LMMStreamChunk.of("", eod=True)

        return generate_stream()

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        elements = []
        async for element in await Conversation.completion(
            input="What's the weather in Paris?",
            memory=None,
            tools=[get_weather],
            stream=True,
        ):
            elements.append(element)

        # Should have chunks and tool events
        chunks = [e for e in elements if isinstance(e, ConversationMessageChunk)]
        events = [e for e in elements if isinstance(e, ConversationEvent)]

        assert len(events) >= 2  # tool.call started and completed
        assert any(e.category == "tool.call" for e in events)

        # Verify content
        full_content = "".join(
            extract_text_content(chunk.content) for chunk in chunks if chunk.content
        )
        assert "Let me check the weather" in full_content or "sunny" in full_content.lower()


@mark.asyncio
async def test_conversation_completion_with_memory_callback():
    """Test conversation completion with memory remember callback."""
    remembered_messages = []

    async def remember_callback(*messages: ConversationMessage):
        remembered_messages.extend(messages)

    async def recall_callback():
        return []

    memory = ConversationMemory(
        recall=recall_callback,
        remember=remember_callback,
    )

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("Hello!")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        input_msg = ConversationMessage.user("Hi")
        await Conversation.completion(
            input=input_msg,
            memory=memory,
            tools=None,
        )

        # Check that both input and response were remembered
        assert len(remembered_messages) == 2
        assert remembered_messages[0].role == "user"
        assert extract_text_content(remembered_messages[0].content) == "Hi"
        assert remembered_messages[1].role == "model"
        assert extract_text_content(remembered_messages[1].content) == "Hello!"


@mark.asyncio
async def test_conversation_completion_tool_error_handling():
    """Test conversation completion handles tool errors gracefully."""
    call_count = 0

    @tool
    async def failing_tool() -> str:
        """A tool that always fails."""
        raise ValueError("Tool error!")

    tool_request = LMMToolRequest(
        identifier=str(uuid4()),
        tool="failing_tool",
        arguments={},
    )

    async def mock_completion(**kwargs) -> LMMCompletion | LMMToolRequests:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LMMToolRequests.of([tool_request])
        else:
            # After seeing the tool error, provide a response
            return LMMCompletion.of("I encountered an error with the tool.")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            input="Use the failing tool",
            memory=None,
            tools=[failing_tool],
        )

        assert "error" in extract_text_content(result.content).lower()
        assert call_count == 2


@mark.asyncio
async def test_conversation_message_from_multimodal():
    """Test creating conversation message from various multimodal inputs."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("Received your message")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        # Test with string input
        result = await Conversation.completion(
            input="Simple string",
            memory=None,
            tools=None,
        )
        assert isinstance(result, ConversationMessage)

        # Test with MultimodalContent input
        result = await Conversation.completion(
            input=MultimodalContent.of("Content object"),
            memory=None,
            tools=None,
        )
        assert isinstance(result, ConversationMessage)

        # Test with ConversationMessage input
        msg = ConversationMessage.user("Direct message")
        result = await Conversation.completion(
            input=msg,
            memory=None,
            tools=None,
        )
        assert isinstance(result, ConversationMessage)


@mark.asyncio
async def test_conversation_completion_empty_response():
    """Test handling of empty LMM responses."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            input="Give me nothing",
            memory=None,
            tools=None,
        )

        assert isinstance(result, ConversationMessage)
        assert extract_text_content(result.content) == ""
        assert result.role == "model"


@mark.asyncio
async def test_conversation_completion_with_toolbox():
    """Test conversation completion with Toolbox instead of tool list."""

    @tool
    async def tool1() -> str:
        return "tool1"

    @tool
    async def tool2() -> str:
        return "tool2"

    toolbox = Toolbox.of(tool1, tool2)

    async def mock_completion(
        *,
        tools=None,
        **kwargs,
    ) -> LMMCompletion:
        # Verify tools were passed
        assert tools is not None
        return LMMCompletion.of("Tools are available")

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        result = await Conversation.completion(
            input="What tools do you have?",
            memory=None,
            tools=toolbox,
        )

        assert extract_text_content(result.content) == "Tools are available"


@mark.asyncio
async def test_conversation_stream_message_identifier_consistency():
    """Test that all chunks in a stream share the same message identifier."""

    async def mock_completion(
        *,
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamChunk]:
        if not stream:
            raise ValueError("Expected streaming")

        async def generate_stream():
            yield LMMStreamChunk.of("Part 1 ")
            yield LMMStreamChunk.of("Part 2 ")
            yield LMMStreamChunk.of("Part 3", eod=True)

        return generate_stream()

    async with ctx.scope("test", LMM(completing=mock_completion), Conversation()):
        message_ids = set()
        async for element in await Conversation.completion(
            input="Stream test",
            memory=None,
            tools=None,
            stream=True,
        ):
            if isinstance(element, ConversationMessageChunk):
                message_ids.add(element.message_identifier)

        # All chunks should have the same message ID
        assert len(message_ids) == 1
