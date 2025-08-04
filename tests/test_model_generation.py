from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from pytest import mark, raises

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
from draive.generation.model import ModelGeneration
from draive.lmm import LMMToolRequests
from draive.parameters import DataModel


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


# Test data models
class Person(DataModel):
    name: str
    age: int
    email: str | None = None


class Product(DataModel):
    name: str
    price: float
    category: str
    in_stock: bool = True


class ComplexModel(DataModel):
    title: str
    items: Sequence[str]


@mark.asyncio
async def test_model_generation_simple():
    """Test basic model generation with simple data model."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        context: Sequence[LMMContextElement],
        output: Any = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.call_count += 1
        tracker.last_context = context
        tracker.last_instruction = instruction
        tracker.last_output = output

        # Return valid JSON for Person model
        json_response = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person with the given details",
            input="Create a 30-year-old software engineer named John Doe",
            schema_injection="skip",
        )

        assert isinstance(result, Person)
        assert result.name == "John Doe"
        assert result.age == 30
        assert result.email == "john@example.com"
        assert tracker.call_count == 1
        assert tracker.last_output == Person


@mark.asyncio
async def test_model_generation_with_schema_injection_full():
    """Test model generation with full schema injection."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_instruction = instruction
        # With full schema injection, instruction should be processed differently
        # Let's just verify the instruction was received and contains our base instruction
        assert "generate a product" in instruction.lower()

        json_response = '{"name": "Test Product", "price": 99.99, "category": "Electronics"}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Product,
            instruction="Generate a product",
            input="Create an electronics product",
            schema_injection="full",
        )

        assert isinstance(result, Product)
        assert result.name == "Test Product"
        assert result.price == 99.99
        assert result.category == "Electronics"
        assert result.in_stock is True  # default value


@mark.asyncio
async def test_model_generation_with_schema_injection_simplified():
    """Test model generation with simplified schema injection."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_instruction = instruction
        json_response = '{"name": "Simple Product", "price": 19.99, "category": "Books"}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Product,
            instruction="Generate a product",
            input="Create a book product",
            schema_injection="simplified",
        )

        assert isinstance(result, Product)
        assert result.name == "Simple Product"


@mark.asyncio
async def test_model_generation_with_schema_injection_skip():
    """Test model generation with schema injection skipped."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        instruction: str | None = None,
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_instruction = instruction
        # Instruction should not contain schema when skipped
        assert "FORMAT" not in instruction and "SCHEMA" not in instruction

        json_response = '{"name": "No Schema Product", "price": 5.99, "category": "Other"}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Product,
            instruction="Generate a product",
            input="Create any product",
            schema_injection="skip",
        )

        assert isinstance(result, Product)
        assert result.name == "No Schema Product"


@mark.asyncio
async def test_model_generation_with_tools():
    """Test model generation with tool usage."""
    call_count = 0

    @tool
    async def get_product_info(product_id: str) -> dict[str, Any]:
        """Get product information by ID."""
        return {
            "name": f"Product {product_id}",
            "price": 29.99,
            "category": "Tools",
            "in_stock": True,
        }

    tool_request = LMMToolRequest(
        identifier=str(uuid4()),
        tool="get_product_info",
        arguments={"product_id": "123"},
    )

    async def mock_completion(**kwargs) -> LMMCompletion | LMMToolRequests:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call requests tool usage
            return LMMToolRequests.of([tool_request])
        else:
            # After tool response, return final result
            json_response = '{"name": "Product 123", "price": 29.99, "category": "Tools"}'
            return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Product,
            instruction="Generate a product using available tools",
            input="Get product information for ID 123",
            tools=[get_product_info],
            schema_injection="skip",
        )

        assert isinstance(result, Product)
        assert result.name == "Product 123"
        assert result.price == 29.99
        assert result.category == "Tools"
        assert call_count == 2


@mark.asyncio
async def test_model_generation_with_examples():
    """Test model generation with examples."""
    tracker = MockLMMTracker()

    async def mock_completion(
        *,
        context: Sequence[LMMContextElement],
        **extra: Any,
    ) -> LMMCompletion:
        tracker.last_context = context
        # Context should include examples (input + completion pairs)
        assert len(context) >= 3  # 2 examples + 1 input

        json_response = '{"name": "Generated Person", "age": 25}'
        return LMMCompletion.of(json_response)

    examples = [
        ("Create a young person", Person(name="Alice", age=22)),
        ("Create an older person", Person(name="Bob", age=45, email="bob@test.com")),
    ]

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input="Create a person aged 25",
            examples=examples,
            schema_injection="skip",
        )

        assert isinstance(result, Person)
        assert result.name == "Generated Person"
        assert result.age == 25


@mark.asyncio
async def test_model_generation_with_custom_decoder():
    """Test model generation with custom decoder."""

    def custom_decoder(content: MultimodalContent) -> Person:
        # Custom decoder that processes non-JSON content
        text = extract_text_content(content)
        if "CUSTOM:" in text:
            parts = text.replace("CUSTOM:", "").strip().split("|")
            return Person(
                name=parts[0], age=int(parts[1]), email=parts[2] if len(parts) > 2 else None
            )
        else:
            return Person.from_json(text)

    async def mock_completion(**kwargs) -> LMMCompletion:
        # Return custom format instead of JSON
        return LMMCompletion.of("CUSTOM:Jane Doe|28|jane@example.com")

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input="Create a custom format person",
            decoder=custom_decoder,
            schema_injection="skip",
        )

        assert isinstance(result, Person)
        assert result.name == "Jane Doe"
        assert result.age == 28
        assert result.email == "jane@example.com"


@mark.asyncio
async def test_model_generation_json_parsing_error():
    """Test model generation handles JSON parsing errors."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        # Return invalid JSON
        return LMMCompletion.of("This is not valid JSON at all")

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        with raises(ValueError):  # Should raise JSON parsing error
            await ModelGeneration.generate(
                Person,
                instruction="Generate a person",
                input="Create a person",
            )


@mark.asyncio
async def test_model_generation_complex_model():
    """Test model generation with complex nested data model."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        json_response = """{
            "title": "Complex Example",
            "items": ["item1", "item2", "item3"]
        }"""
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            ComplexModel,
            instruction="Generate a complex model",
            input="Create a complex data structure",
            schema_injection="skip",
        )

        assert isinstance(result, ComplexModel)
        assert result.title == "Complex Example"
        assert len(result.items) == 3
        assert result.items[0] == "item1"


@mark.asyncio
async def test_model_generation_with_toolbox():
    """Test model generation with Toolbox instead of tool list."""

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
        json_response = '{"name": "Toolbox Product", "price": 15.99, "category": "Test"}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Product,
            instruction="Generate a product",
            input="Create a product using toolbox",
            schema_injection="skip",
            tools=toolbox,
        )

        assert isinstance(result, Product)
        assert result.name == "Toolbox Product"


@mark.asyncio
async def test_model_generation_instruction_types():
    """Test model generation with different instruction types."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        json_response = '{"name": "Instruction Test", "age": 35}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        # Test with string instruction
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input="Create a person",
            schema_injection="skip",
        )
        assert isinstance(result, Person)

        # Test with Instruction object
        from draive import Instruction

        instruction_obj = Instruction.of("Generate a person with proper formatting")
        result = await ModelGeneration.generate(
            Person,
            instruction=instruction_obj,
            input="Create a person",
            schema_injection="skip",
        )
        assert isinstance(result, Person)


@mark.asyncio
async def test_model_generation_tool_error_handling():
    """Test model generation handles tool errors gracefully."""
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
            json_response = '{"name": "Error Recovery", "age": 0}'
            return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input="Use the failing tool",
            tools=[failing_tool],
            schema_injection="skip",
        )

        assert isinstance(result, Person)
        assert result.name == "Error Recovery"
        assert call_count == 2


@mark.asyncio
async def test_model_generation_empty_examples():
    """Test model generation with empty examples list."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        json_response = '{"name": "No Examples", "age": 40}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input="Create a person without examples",
            examples=[],  # Empty examples
            schema_injection="skip",
        )

        assert isinstance(result, Person)
        assert result.name == "No Examples"


@mark.asyncio
async def test_model_generation_multimodal_input():
    """Test model generation with different input types."""

    async def mock_completion(**kwargs) -> LMMCompletion:
        json_response = '{"name": "Multimodal Test", "age": 33}'
        return LMMCompletion.of(json_response)

    async with ctx.scope("test", LMM(completing=mock_completion), ModelGeneration()):
        # Test with string input
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input="Simple string input",
            schema_injection="skip",
        )
        assert isinstance(result, Person)

        # Test with MultimodalContent input
        result = await ModelGeneration.generate(
            Person,
            instruction="Generate a person",
            input=MultimodalContent.of("Multimodal content input"),
            schema_injection="skip",
        )
        assert isinstance(result, Person)
