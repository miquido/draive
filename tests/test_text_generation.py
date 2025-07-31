from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from pytest import mark

from draive import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMToolRequest,
    MultimodalContent,
    Toolbox,
    ctx,
    tool,
)
from draive.generation.text import TextGeneration
from draive.lmm import LMMToolRequests


class MockLMMTracker:
    """Tracks LMM calls for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_context = None
        self.last_instruction = None
        self.last_tools = None
        self.last_output = None


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
async def test_text_generation_simple():
    """Test basic text generation."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        context: Sequence[LMMContextElement],
        output: str = "text",
        **extra: Any,
    ) -> LMMCompletion:
        tracker.call_count += 1
        tracker.last_context = context
        tracker.last_instruction = instruction
        tracker.last_output = output

        return LMMCompletion.of("This is a generated text response.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate a text response",
            input="Create some text about AI",
        )

        assert isinstance(result, str)
        assert result == "This is a generated text response."
        assert tracker.call_count == 1
        assert tracker.last_output == "text"


@mark.asyncio
async def test_text_generation_without_instruction():
    """Test text generation without explicit instruction."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_instruction = instruction
        assert instruction is None
        return LMMCompletion.of("Generated without specific instruction.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            input="Generate some text",
        )

        assert result == "Generated without specific instruction."


@mark.asyncio
async def test_text_generation_with_tools():
    """Test text generation with tool usage."""
    call_count = 0

    @tool
    async def get_weather(city: str) -> str:
        """Get weather information for a city."""
        return f"The weather in {city} is sunny and 25°C."

    tool_request = LMMToolRequest(
        identifier=str(uuid4()),
        tool="get_weather",
        arguments={"city": "Paris"},
    )

    async def mock_completion(**kwargs) -> LMMCompletion | LMMToolRequests:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call requests tool usage
            return LMMToolRequests.of([tool_request])
        else:
            # After tool response, return final result
            return LMMCompletion.of(
                "Based on the weather data, it's a beautiful sunny day in Paris with 25°C."
            )

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate weather report",
            input="Tell me about the weather in Paris",
            tools=[get_weather],
        )

        assert isinstance(result, str)
        assert "sunny" in result.lower()
        assert "paris" in result.lower()
        assert call_count == 2


@mark.asyncio
async def test_text_generation_with_examples():
    """Test text generation with examples."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        context: Sequence[LMMContextElement],
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_context = context
        # Context should include examples (input + completion pairs)
        assert len(context) >= 5  # 2 examples (4 messages) + 1 input
        return LMMCompletion.of("Generated text following the examples pattern.")

    examples = [
        ("Write about cats", "Cats are independent and graceful animals."),
        ("Write about dogs", "Dogs are loyal and friendly companions."),
    ]

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate descriptive text",
            input="Write about birds",
            examples=examples,
        )

        assert isinstance(result, str)
        assert result == "Generated text following the examples pattern."


@mark.asyncio
async def test_text_generation_with_toolbox():
    """Test text generation with Toolbox instead of tool list."""

    @tool
    async def tool1() -> str:
        return "tool1_result"

    @tool
    async def tool2() -> str:
        return "tool2_result"

    toolbox = Toolbox.of(tool1, tool2)

    async def mock_completion(
        *,
        tools=None,
        **kwargs,
    ) -> LMMCompletion:
        # Verify tools were passed
        assert tools is not None
        return LMMCompletion.of("Generated text using available tools.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text",
            input="Create text using toolbox",
            tools=toolbox,
        )

        assert isinstance(result, str)
        assert result == "Generated text using available tools."


@mark.asyncio
async def test_text_generation_multimodal_input():
    """Test text generation with different input types."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("Generated text from multimodal input.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        # Test with string input
        result = await TextGeneration.generate(
            instruction="Generate text",
            input="Simple string input",
        )
        assert isinstance(result, str)
        assert result == "Generated text from multimodal input."

        # Test with MultimodalContent input
        result = await TextGeneration.generate(
            instruction="Generate text",
            input=MultimodalContent.of("Multimodal content input"),
        )
        assert isinstance(result, str)
        assert result == "Generated text from multimodal input."


@mark.asyncio
async def test_text_generation_instruction_types():
    """Test text generation with different instruction types."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("Generated with instruction object.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        # Test with string instruction
        result = await TextGeneration.generate(
            instruction="Generate text with string instruction",
            input="Create some text",
        )
        assert isinstance(result, str)

        # Test with Instruction object
        from draive import Instruction

        instruction_obj = Instruction.of("Generate text with instruction object")
        result = await TextGeneration.generate(
            instruction=instruction_obj,
            input="Create some text",
        )
        assert isinstance(result, str)
        assert result == "Generated with instruction object."


@mark.asyncio
async def test_text_generation_tool_error_handling():
    """Test text generation handles tool errors gracefully."""
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
            return LMMCompletion.of(
                "I encountered an error with the tool, but here's a fallback response."
            )

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text",
            input="Use the failing tool",
            tools=[failing_tool],
        )

        assert isinstance(result, str)
        assert "error" in result.lower() or "fallback" in result.lower()
        assert call_count == 2


@mark.asyncio
async def test_text_generation_empty_examples():
    """Test text generation with empty examples list."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("Generated without examples.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text",
            input="Create text without examples",
            examples=[],  # Empty examples
        )

        assert isinstance(result, str)
        assert result == "Generated without examples."


@mark.asyncio
async def test_text_generation_empty_response():
    """Test text generation handles empty responses."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of("")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text",
            input="Return empty response",
        )

        assert isinstance(result, str)
        assert result == ""


@mark.asyncio
async def test_text_generation_long_response():
    """Test text generation with longer text response."""

    long_text = """This is a longer text response that spans multiple sentences.
    It demonstrates how the text generation can handle extended content.
    The response includes various types of information and maintains coherence
    throughout the entire generated text."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of(long_text)

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate a long text response",
            input="Create extended content about text generation",
        )

        assert isinstance(result, str)
        assert len(result) > 100  # Ensure it's actually long
        assert "longer text response" in result
        assert "coherence" in result


@mark.asyncio
async def test_text_generation_complex_instruction():
    """Test text generation with complex multi-part instruction."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        **kwargs,
    ) -> LMMCompletion:
        tracker.last_instruction = instruction
        return LMMCompletion.of("Generated following complex instructions.")

    complex_instruction = """
    Generate a text response that:
    1. Uses formal language
    2. Includes specific technical terms
    3. Maintains professional tone
    4. Provides actionable insights
    """

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction=complex_instruction,
            input="Create professional analysis text",
        )

        assert isinstance(result, str)
        assert result == "Generated following complex instructions."
        # Verify the complex instruction was passed through
        assert "formal language" in tracker.last_instruction


@mark.asyncio
async def test_text_generation_special_characters():
    """Test text generation with special characters and formatting."""

    special_text = "Generated text with special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of(special_text)

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text with special characters",
            input="Include various symbols",
        )

        assert isinstance(result, str)
        assert result == special_text
        assert "@#$%^&*()" in result


@mark.asyncio
async def test_text_generation_unicode_content():
    """Test text generation with Unicode characters."""

    unicode_text = "Generated text with Unicode: 你好 世界 🌍 ñáéíóú àèìòù äëïöü ß"

    async def mock_completion(**kwargs) -> LMMCompletion:
        return LMMCompletion.of(unicode_text)

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text with Unicode characters",
            input="Include international characters",
        )

        assert isinstance(result, str)
        assert result == unicode_text
        assert "你好" in result
        assert "🌍" in result
        assert "ñáéíóú" in result


@mark.asyncio
async def test_text_generation_context_preservation():
    """Test that context is properly passed to LMM."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        context: Sequence[LMMContextElement],
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_context = context
        # Should have at least the input context
        assert len(context) >= 1
        return LMMCompletion.of("Context preserved correctly.")

    async with ctx.scope("test", LMM(completing=mock_completion), TextGeneration()):
        result = await TextGeneration.generate(
            instruction="Generate text",
            input="Test context preservation",
        )

        assert result == "Context preserved correctly."
        assert tracker.last_context is not None
        assert len(tracker.last_context) >= 1
