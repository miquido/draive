from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from pytest import mark

from draive import (
    GenerativeModel,
    MultimodalContent,
    Toolbox,
    ctx,
    tool,
)
from draive.conversation import Conversation, ConversationMessage
from draive.models import (
    ModelContextElement,
    ModelInput,
    ModelOutput,
    ModelToolRequest,
    ModelToolsDeclaration,
)
from draive.utils import Memory as ConversationMemory


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

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        tracker.call_count += 1
        tracker.last_context = context
        tracker.last_instruction = instructions
        tracker.last_tools = tools
        return ModelOutput.of(MultimodalContent.of("Hello! How can I help you?"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
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

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        tracker.call_count += 1
        tracker.last_context = context
        return ModelOutput.of(MultimodalContent.of("I can see we talked before."))

    memory_messages = [
        ConversationMessage.user("Previous message"),
        ConversationMessage.model("Previous response"),
    ]

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
        result = await Conversation.completion(
            input="Do you remember our conversation?",
            memory=memory_messages,
            tools=None,
        )

        assert isinstance(result, ConversationMessage)
        assert extract_text_content(result.content) == "I can see we talked before."
        assert tracker.call_count == 1
        # Check that memory+input were included (context mutates to include output)
        assert len(tracker.last_context or []) >= 3


@mark.asyncio
async def test_conversation_completion_with_instruction():
    """Test conversation completion with custom instruction."""
    tracker = MockLMMTracker()

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        tracker.last_instruction = instructions
        return ModelOutput.of(MultimodalContent.of("I'm a helpful assistant!"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
        result = await Conversation.completion(
            instructions="You are a helpful assistant",
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

    tool_request = ModelToolRequest.of(
        str(uuid4()),
        tool="calculate",
        arguments={"a": 5, "b": 3},
    )

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call requests tool usage
            return ModelOutput.of(tool_request)
        else:
            # After tool response, return final result
            return ModelOutput.of(MultimodalContent.of("The sum is 8"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
        result = await Conversation.completion(
            input="What is 5 + 3?",
            memory=None,
            tools=[calculate],
        )

        assert extract_text_content(result.content) == "The sum is 8"
        assert call_count == 2  # Initial request + after tool response

    # Tool events streaming behavior changed; streaming path is provider-dependent


@mark.asyncio
async def test_conversation_completion_with_memory_callback():
    """Test conversation completion with memory remember callback."""
    remembered_messages: list[ConversationMessage] = []

    async def remember_callback(*elements: ModelContextElement, **_: Any):
        for el in elements:
            if isinstance(el, ModelInput):
                remembered_messages.append(ConversationMessage.user(el.content))
            else:
                remembered_messages.append(ConversationMessage.model(el.content))

    async def recall_callback(**_: Any):
        from draive.models import ModelMemoryRecall

        return ModelMemoryRecall.empty

    memory = ConversationMemory(
        recalling=recall_callback,
        remembering=remember_callback,
    )

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Hello!"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
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

    tool_request = ModelToolRequest.of(
        str(uuid4()),
        tool="failing_tool",
    )

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelOutput.of(tool_request)
        else:
            # After seeing the tool error, provide a response
            return ModelOutput.of(MultimodalContent.of("I encountered an error with the tool."))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
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

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Received your message"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
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

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of(""))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
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

    async def mock_generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[ModelContextElement],
        output: str | type | None = None,
        **kwargs,
    ) -> ModelOutput:
        # Verify tools were passed
        assert tools is not None and bool(tools)
        return ModelOutput.of(MultimodalContent.of("Tools are available"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating), Conversation()):
        result = await Conversation.completion(
            input="Example",
            memory=None,
            tools=toolbox,
        )

        assert extract_text_content(result.content) == "Tools are available"

    # Skipped provider-dependent streaming identifier consistency
