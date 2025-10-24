from collections.abc import Sequence
from uuid import uuid4

from haiway import MissingState, State, ctx
from pytest import mark, raises

from draive import MultimodalContent, tool
from draive.evaluation import EvaluatorResult
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelOutput,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.parameters import DataModel
from draive.stages import Stage
from draive.stages.types import (
    StageException,
    StageState,
)
from draive.utils import Memory

# Use Memory.volatile for testing instead of custom mock


class MockModelTracker:
    """Tracks model calls for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_context = None
        self.last_instructions = None
        self.last_tools = None
        self.last_output = None


class ExampleData(DataModel):
    value: str
    count: int = 0


class ExampleState(State):
    data: str = "test"


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


# simple in-memory Memory helper for StageState
def make_volatile_memory(initial: StageState) -> Memory[StageState, StageState]:
    store = {"value": initial}

    async def recall(**extra):
        return store["value"]

    async def remember(*items: StageState, **extra):
        if items:
            store["value"] = items[-1]

    return Memory(recalling=recall, remembering=remember)


def make_evaluator_result(*, score: float, threshold: float) -> EvaluatorResult:
    return EvaluatorResult.of(
        "mock_evaluator",
        score=score,
        threshold=threshold,
    )


@mark.asyncio
async def test_stage_predefined_simple():
    """Test basic predefined stage creation and execution."""
    # According to Stage.predefined logic:
    # - First element becomes ModelInput
    # - elements[0] (even index) becomes ModelInput
    # - elements[1] (odd index) becomes ModelOutput
    stage = Stage.predefined(
        "User input",
        "Assistant response",
    )

    initial_state = StageState.of(context=(), result=MultimodalContent.of("initial"))

    result_state = await stage(state=initial_state)

    # Should have extended context with input and output
    assert len(result_state.context) == 2
    assert isinstance(result_state.context[0], ModelInput)
    assert isinstance(result_state.context[1], ModelOutput)

    # Result remains unchanged for predefined stage unless result is provided
    assert extract_text_content(result_state.result) == "initial"


@mark.asyncio
async def test_stage_predefined_with_existing_context():
    """Test predefined stage with existing context."""
    existing_context = (
        ModelInput.of(MultimodalContent.of("Previous input")),
        ModelOutput.of(MultimodalContent.of("Previous completion")),
    )

    stage = Stage.predefined(
        "New input",
        "New completion",
    )

    initial_state = StageState.of(context=existing_context, result=MultimodalContent.of("initial"))

    result_state = await stage(state=initial_state)

    # Should have extended existing context by 2 more elements
    assert len(result_state.context) == 4
    assert result_state.context[:2] == existing_context
    # Result remains unchanged for predefined stage unless result is provided
    assert extract_text_content(result_state.result) == "initial"


@mark.asyncio
async def test_stage_completion_basic():
    """Test basic completion stage."""
    tracker = MockModelTracker()

    async def mock_generating(
        *,
        instructions=None,
        context: Sequence[ModelContextElement],
        tools=None,
        output=None,
        stream=False,
        **kwargs,
    ) -> ModelOutput:
        tracker.call_count += 1
        tracker.last_context = context
        tracker.last_instructions = instructions
        tracker.last_tools = tools
        tracker.last_output = output

        return ModelOutput.of(MultimodalContent.of("Completion response"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.completion("What is AI?", instructions="Explain clearly")

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await stage(state=initial_state)

        assert tracker.call_count == 1
        assert tracker.last_instructions is not None
        assert len(result_state.context) == 2  # input + completion
        assert extract_text_content(result_state.result) == "Completion response"


@mark.asyncio
async def test_stage_completion_with_tools():
    """Test completion stage with tool usage."""
    call_count = 0

    @tool
    async def test_tool(query: str) -> str:
        """Test tool for completion."""
        return f"Tool result for: {query}"

    tool_request = ModelToolRequest.of(
        str(uuid4()),
        tool="test_tool",
        arguments={"query": "test"},
    )

    async def mock_generating(**kwargs) -> ModelOutput:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelOutput.of(tool_request)
        else:
            return ModelOutput.of(MultimodalContent.of("Final completion with tool result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.completion("Use the tool", tools=[test_tool])

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await stage(state=initial_state)

        assert call_count == 2
        # Context should include input, tool requests, tool responses, and completion
        assert len(result_state.context) >= 4
        assert extract_text_content(result_state.result) == "Final completion with tool result"


@mark.asyncio
async def test_stage_prompting_completion():
    """Test prompting completion stage."""

    async def get_dynamic_input():
        return "Dynamic input from function"

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Response to dynamic input"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.prompting_completion(
            get_dynamic_input, instructions="Process the dynamic input"
        )

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await stage(state=initial_state)

        assert len(result_state.context) == 2
        assert extract_text_content(result_state.result) == "Response to dynamic input"


@mark.asyncio
async def test_stage_loopback_completion():
    """Test loopback completion stage."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Refined response"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.loopback_completion(instructions="Refine the previous response")

        # Initial state with existing context ending in completion
        initial_context = (
            ModelInput.of(MultimodalContent.of("Original input")),
            ModelOutput.of(MultimodalContent.of("Original completion")),
        )

        initial_state = StageState.of(
            context=initial_context, result=MultimodalContent.of("Original completion")
        )

        result_state = await stage(state=initial_state)

        # Should have new completion using last completion as input
        assert len(result_state.context) == 2  # new input (from last completion) + new completion
        assert extract_text_content(result_state.result) == "Refined response"


@mark.asyncio
async def test_stage_loopback_completion_invalid_context():
    """Test loopback completion with invalid context (no completion)."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Default response"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.loopback_completion(instructions="Refine response")

        # Empty context (valid but no completion to loop back)
        initial_state = StageState.of(context=(), result=MultimodalContent.of("result"))

        result_state = await stage(state=initial_state)

        # Should return original state unchanged when context is invalid/empty
        assert result_state == initial_state


@mark.asyncio
async def test_stage_result_completion():
    """Test result completion stage."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Processed result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.result_completion(instructions="Process the current result")

        initial_state = StageState.of(
            context=(), result=MultimodalContent.of("Current result content")
        )

        result_state = await stage(state=initial_state)

        assert len(result_state.context) == 2  # input (from result) + completion
        assert extract_text_content(result_state.result) == "Processed result"


@mark.asyncio
async def test_stage_transform_result():
    """Test result transformation stage."""

    async def transform_result(result: MultimodalContent) -> MultimodalContent:
        return MultimodalContent.of("Transformed: ", result)

    stage = Stage.transform_result(transform_result)

    initial_state = StageState.of(
        context=(
            ModelInput.of(MultimodalContent.of("test")),
            ModelOutput.of(MultimodalContent.of("completion")),
        ),
        result=MultimodalContent.of("original result"),
    )

    result_state = await stage(state=initial_state)

    # Context should remain unchanged
    assert result_state.context == initial_state.context
    # Result should be transformed
    assert extract_text_content(result_state.result) == "Transformed: original result"


@mark.asyncio
async def test_stage_transform_context():
    """Test context transformation stage."""

    async def transform_context(context: ModelContext) -> ModelContext:
        return (
            *context,
            ModelInput.of(MultimodalContent.of("Added input")),
            ModelOutput.of(MultimodalContent.of("Added completion")),
        )

    stage = Stage.transform_context(transform_context)

    initial_context = (
        ModelInput.of(MultimodalContent.of("original")),
        ModelOutput.of(MultimodalContent.of("original completion")),
    )
    initial_state = StageState.of(context=initial_context, result=MultimodalContent.of("result"))

    result_state = await stage(state=initial_state)

    # Context should be transformed - original 2 + added 2 = 4
    assert len(result_state.context) == 4
    assert result_state.context[:2] == initial_context
    # Result should remain unchanged
    assert result_state.result == initial_state.result


@mark.asyncio
async def test_stage_trim_context():
    """Test context trimming stage."""
    initial_context = (
        ModelInput.of(MultimodalContent.of("input1")),
        ModelOutput.of(MultimodalContent.of("completion1")),
        ModelInput.of(MultimodalContent.of("input2")),
        ModelOutput.of(MultimodalContent.of("completion2")),
        ModelInput.of(MultimodalContent.of("input3")),
        ModelOutput.of(MultimodalContent.of("completion3")),
    )

    initial_state = StageState.of(context=initial_context, result=MultimodalContent.of("result"))

    # Test trimming to first 2 elements
    trim_stage = Stage.trim_context(limit=2)
    result_state = await trim_stage(state=initial_state)

    assert len(result_state.context) == 2
    assert result_state.context == initial_context[:2]
    assert result_state.result == initial_state.result

    # Test clearing context
    clear_stage = Stage.trim_context(limit=None)
    result_state = await clear_stage(state=initial_state)

    assert len(result_state.context) == 0
    assert result_state.result == initial_state.result


@mark.asyncio
async def test_stage_strip_context_tools():
    """Test stripping tool calls from context."""
    tool_request = ModelToolRequest.of(
        "test_id",
        tool="test_tool",
        arguments={},
    )
    tool_response = ModelToolResponse.of(
        "test_id",
        tool="test_tool",
        content=MultimodalContent.of("tool result"),
        handling="response",
    )

    initial_context = (
        ModelInput.of(MultimodalContent.of("input")),
        ModelOutput.of(MultimodalContent.of("completion")),
        ModelOutput.of(tool_request),
        ModelInput.of(tool_response),
        ModelInput.of(MultimodalContent.of("another input")),
        ModelOutput.of(MultimodalContent.of("final completion")),
    )

    initial_state = StageState.of(context=initial_context, result=MultimodalContent.of("result"))

    stage = Stage.strip_context_tools()
    result_state = await stage(state=initial_state)

    # Tool request/response blocks are stripped to pure content but context length is preserved
    assert len(result_state.context) == 6
    for element in result_state.context:
        if isinstance(element, ModelInput) or isinstance(element, ModelOutput):
            assert isinstance(element.content, MultimodalContent)


# NOTE: Stage.tool_call tests are commented out because they test an API pattern
# that doesn't work by design. Stage.tool_call creates contexts ending with
# LMMToolResponses, but StageState validation requires contexts to end with
# LMMCompletion. This is intentional - tool_call is meant for advanced usage
# with manual context management, not standalone execution.
#
# The correct usage is within completion stages with tools parameter:
# Stage.completion("Use tool", tools=[my_tool])

# @mark.asyncio
# async def test_stage_tool_call():
#     """Test tool call stage - SKIPPED: Framework design limitation."""
#     pass


# @mark.asyncio
# async def test_stage_tool_call_error():
#     """Test tool call error stage - SKIPPED: Framework design limitation."""
#     pass


@mark.asyncio
async def test_stage_memory_remember():
    """Test memory remember stage."""
    empty_state = StageState.of(context=(), result=MultimodalContent.empty)

    def make_volatile_memory(initial: StageState) -> Memory[StageState, StageState]:
        store = {"value": initial}

        async def recall(**extra):
            return store["value"]

        async def remember(*items: StageState, **extra):
            if items:
                store["value"] = items[-1]

        return Memory(recalling=recall, remembering=remember)

    memory = make_volatile_memory(empty_state)
    stage = Stage.memory_remember(memory)

    initial_state = StageState.of(
        context=(
            ModelInput.of(MultimodalContent.of("test")),
            ModelOutput.of(MultimodalContent.of("completion")),
        ),
        result=MultimodalContent.of("result"),
    )

    result_state = await stage(state=initial_state)

    # State should be unchanged
    assert result_state == initial_state
    # Memory should have stored the state
    stored_state = await memory.recall()
    assert stored_state == initial_state


@mark.asyncio
async def test_stage_memory_recall():
    """Test memory recall stage."""
    # Pre-populate memory
    stored_state = StageState.of(
        context=(
            ModelInput.of(MultimodalContent.of("stored")),
            ModelOutput.of(MultimodalContent.of("stored completion")),
        ),
        result=MultimodalContent.of("stored result"),
    )
    memory = make_volatile_memory(stored_state)

    stage = Stage.memory_recall(memory, handling="replace")

    initial_state = StageState.of(context=(), result=MultimodalContent.empty)

    result_state = await stage(state=initial_state)

    # Should have replaced with stored state
    assert result_state == stored_state


@mark.asyncio
async def test_stage_memory_recall_merge():
    """Test memory recall with merge handling."""
    stored_state = StageState.of(
        context=(
            ModelInput.of(MultimodalContent.of("stored")),
            ModelOutput.of(MultimodalContent.of("stored completion")),
        ),
        result=MultimodalContent.of("stored result"),
    )
    memory = make_volatile_memory(stored_state)

    stage = Stage.memory_recall(memory, handling="merge")

    initial_state = StageState.of(
        context=(
            ModelInput.of(MultimodalContent.of("current")),
            ModelOutput.of(MultimodalContent.of("current completion")),
        ),
        result=MultimodalContent.of("current result"),
    )

    result_state = await stage(state=initial_state)

    # Should have merged contexts and results
    assert len(result_state.context) == 4  # 2 + 2
    assert extract_text_content(result_state.result) == "current resultstored result"


@mark.asyncio
async def test_stage_sequence():
    """Test sequential stage execution."""

    async def mock_generating(**kwargs) -> ModelOutput:
        context = kwargs.get("context", ())
        if len(context) == 1:  # First stage
            return ModelOutput.of(MultimodalContent.of("First response"))
        else:  # Second stage
            return ModelOutput.of(MultimodalContent.of("Second response"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage1 = Stage.completion("First input")
        stage2 = Stage.completion("Second input")

        sequence_stage = Stage.sequence(stage1, stage2)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await sequence_stage(state=initial_state)

        # Should have executed both stages sequentially
        assert len(result_state.context) == 4  # 2 inputs + 2 completions
        assert extract_text_content(result_state.result) == "Second response"


@mark.asyncio
async def test_stage_concurrent():
    """Test concurrent stage execution."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Concurrent response"))

    async def merge_results(*, branches: Sequence[StageState | StageException]) -> StageState:
        # Simple merge - take first successful result
        successful = [b for b in branches if isinstance(b, StageState)]
        if successful:
            return successful[0].updated(result=MultimodalContent.of("Merged result"))
        raise RuntimeError("No successful branches")

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage1 = Stage.completion("Input 1")
        stage2 = Stage.completion("Input 2")

        concurrent_stage = Stage.concurrent(stage1, stage2, merge=merge_results)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await concurrent_stage(state=initial_state)

        assert extract_text_content(result_state.result) == "Merged result"


@mark.asyncio
async def test_stage_loop():
    """Test stage loop execution."""
    iteration_count = 0

    async def mock_generating(**kwargs) -> ModelOutput:
        nonlocal iteration_count
        iteration_count += 1
        return ModelOutput.of(MultimodalContent.of(f"Iteration {iteration_count}"))

    async def loop_condition(*, state: StageState, iteration: int) -> bool:
        return iteration < 2  # Stop after 2 iterations

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        base_stage = Stage.completion("Loop input")
        loop_stage = Stage.loop(
            base_stage, condition=loop_condition, condition_check="after_execution"
        )

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await loop_stage(state=initial_state)

        # Should have executed 3 iterations (0, 1, 2) with after_execution check
        assert iteration_count == 3
        assert extract_text_content(result_state.result) == "Iteration 3"


@mark.asyncio
async def test_stage_router():
    """Test stage routing."""

    async def mock_generating(**kwargs) -> ModelOutput:
        instructions = kwargs.get("instructions", "") or ""  # Handle None instructions
        if "select the most appropriate option" in instructions:
            return ModelOutput.of(MultimodalContent.of("<SELECTION>option1</SELECTION>"))
        else:
            return ModelOutput.of(MultimodalContent.of("Option 1 result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage1 = Stage.completion("Option 1 input").with_meta(
            name="option1", description="First option for routing"
        )
        stage2 = Stage.completion("Option 2 input").with_meta(
            name="option2", description="Second option for routing"
        )

        router_stage = Stage.router(stage1, stage2)

        initial_state = StageState.of(
            context=(), result=MultimodalContent.of("routing decision needed")
        )

        result_state = await router_stage(state=initial_state)

        assert extract_text_content(result_state.result) == "Option 1 result"


@mark.asyncio
async def test_stage_when_condition_true():
    """Test conditional stage execution when condition is true."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Executed"))

    async def true_condition(*, state: StageState) -> bool:
        return True

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        base_stage = Stage.completion("Test input")
        conditional_stage = base_stage.when(true_condition)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await conditional_stage(state=initial_state)

        assert extract_text_content(result_state.result) == "Executed"


@mark.asyncio
async def test_stage_when_condition_false():
    """Test conditional stage execution when condition is false."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Should not execute"))

    async def false_condition(*, state: StageState) -> bool:
        return False

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        base_stage = Stage.completion("Test input")
        conditional_stage = base_stage.when(false_condition)

        initial_state = StageState.of(context=(), result=MultimodalContent.of("original"))

        result_state = await conditional_stage(state=initial_state)

        # Should return unchanged state
        assert result_state == initial_state


@mark.asyncio
async def test_stage_when_with_alternative():
    """Test conditional stage with alternative."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Alternative executed"))

    async def false_condition(*, state: StageState) -> bool:
        return False

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        primary_stage = Stage.completion("Primary input")
        alternative_stage = Stage.completion("Alternative input")

        conditional_stage = primary_stage.when(false_condition, alternative=alternative_stage)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await conditional_stage(state=initial_state)

        assert extract_text_content(result_state.result) == "Alternative executed"


@mark.asyncio
async def test_stage_cached():
    """Test stage caching."""
    call_count = 0

    async def mock_generating(**kwargs) -> ModelOutput:
        nonlocal call_count
        call_count += 1
        return ModelOutput.of(MultimodalContent.of(f"Call {call_count}"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        base_stage = Stage.completion("Test input")
        cached_stage = base_stage.cached(limit=5)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        # First execution
        result1 = await cached_stage(state=initial_state)
        assert call_count == 1
        assert extract_text_content(result1.result) == "Call 1"

        # Second execution with same state should use cache
        result2 = await cached_stage(state=initial_state)
        assert call_count == 1  # Should not increase
        assert extract_text_content(result2.result) == "Call 1"


@mark.asyncio
async def test_stage_with_retry():
    """Test stage with retry behavior."""
    attempt_count = 0

    async def failing_completion(**kwargs) -> ModelOutput:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Connection failed")
        return ModelOutput.of(MultimodalContent.of("Success after retries"))

    async with ctx.scope("test", GenerativeModel(generating=failing_completion)):
        base_stage = Stage.completion("Test input")
        retry_stage = base_stage.with_retry(
            limit=3,
            delay=0.01,  # Short delay for testing
            catching=lambda exc: isinstance(exc, ConnectionError),
        )

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await retry_stage(state=initial_state)

        assert attempt_count == 3
        assert extract_text_content(result_state.result) == "Success after retries"


@mark.asyncio
async def test_stage_with_fallback():
    """Test stage with fallback behavior."""

    # Track which stage is being executed to simulate fallback
    execution_attempt = 0

    async def smart_completion(**kwargs) -> ModelOutput:
        nonlocal execution_attempt
        execution_attempt += 1

        if execution_attempt == 1:
            # Primary stage fails
            raise ValueError("Primary stage failed")
        else:
            # Fallback stage succeeds
            return ModelOutput.of(MultimodalContent.of("Fallback executed"))

    async with ctx.scope("test", GenerativeModel(generating=smart_completion)):
        primary_stage = Stage.completion("Primary input")
        fallback_stage = Stage.completion("Fallback input")

        robust_stage = primary_stage.with_fallback(fallback_stage, catching=ValueError)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await robust_stage(state=initial_state)

        # Should have executed both primary (failed) and fallback (succeeded)
        assert execution_attempt == 2
        assert extract_text_content(result_state.result) == "Fallback executed"


@mark.asyncio
async def test_stage_with_volatile_context():
    """Test stage with volatile context."""

    async def mock_completion(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("New result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_completion)):
        volatile_stage = Stage.completion("Test input").with_volatile_context()

        initial_context = (
            ModelInput.of(MultimodalContent.of("existing")),
            ModelOutput.of(MultimodalContent.of("existing completion")),
        )
        initial_state = StageState.of(
            context=initial_context, result=MultimodalContent.of("initial")
        )

        result_state = await volatile_stage(state=initial_state)

        # Context should be reverted to original
        assert result_state.context == initial_context
        # But result should be updated
        assert extract_text_content(result_state.result) == "New result"


@mark.asyncio
async def test_stage_with_volatile_tools_context():
    """Test stage with volatile tools context."""

    @tool
    async def test_tool() -> str:
        return "tool result"

    tool_request = ModelToolRequest.of(
        str(uuid4()),
        tool="test_tool",
        arguments={},
    )

    call_count = 0

    async def mock_completion(**kwargs) -> ModelOutput:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelOutput.of(tool_request)
        return ModelOutput.of(MultimodalContent.of("Final result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_completion)):
        stage = Stage.completion("Use tools", tools=[test_tool]).with_volatile_tools_context()

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await stage(state=initial_state)

        # Should have input, tool request/response and completion in context
        assert len(result_state.context) == 4
    for element in result_state.context:
        if isinstance(element, ModelInput) or isinstance(element, ModelOutput):
            assert isinstance(element.content, MultimodalContent)


@mark.asyncio
async def test_stage_ignore_result():
    """Test stage that ignores its result."""

    async def mock_completion(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("New result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_completion)):
        ignore_stage = Stage.completion("Test input").ignore_result()

        initial_state = StageState.of(context=(), result=MultimodalContent.of("original result"))

        result_state = await ignore_stage(state=initial_state)

        # Result should remain unchanged
        assert result_state.result == initial_state.result
        # But context should be updated
        assert len(result_state.context) == 2


@mark.asyncio
async def test_stage_extend_result():
    """Test stage that extends result."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Extended content"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        extend_stage = Stage.completion("Test input").extend_result()

        initial_state = StageState.of(context=(), result=MultimodalContent.of("Original content"))

        result_state = await extend_stage(state=initial_state)

        # Result should be extended
        result_text = extract_text_content(result_state.result)
        assert "Original content" in result_text
        assert "Extended content" in result_text


@mark.asyncio
async def test_stage_with_meta():
    """Test stage metadata operations."""
    stage = Stage.completion("Test input")

    # Add metadata
    meta_stage = stage.with_meta(
        name="test_stage", description="A test stage", tags=["test", "example"]
    )

    assert meta_stage.meta.name == "test_stage"
    assert meta_stage.meta.description == "A test stage"
    assert meta_stage.meta.tags == ("test", "example")

    # Original stage should be unchanged
    assert stage.meta.name is None


@mark.asyncio
async def test_stage_execute():
    """Test stage execute method."""

    async def mock_generating(**kwargs) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of("Execution result"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.completion("Test input")

        # Execute with custom initial values
        result = await stage.execute(
            ExampleData(value="test", count=5), memory=(), result=MultimodalContent.of("initial")
        )

        assert extract_text_content(result) == "Execution result"


@mark.asyncio
async def test_stage_noop():
    """Test noop stage."""
    initial_state = StageState.of(context=(), result=MultimodalContent.of("unchanged"))

    result_state = await Stage.noop(state=initial_state)

    # Should return exactly the same state
    assert result_state == initial_state


@mark.asyncio
async def test_stage_exception_handling():
    """Test stage exception handling."""

    async def failing_completion(**kwargs) -> ModelOutput:
        raise RuntimeError("Completion failed")

    async with ctx.scope("test", GenerativeModel(generating=failing_completion)):
        stage = Stage.completion("Test input")

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        with raises(StageException) as exc_info:
            await stage(state=initial_state)

        # Should wrap the original exception in StageException
        assert "Stage execution failed" in str(exc_info.value)
        assert exc_info.value.state == initial_state


@mark.asyncio
async def test_stage_result_evaluation():
    """Test stage with result evaluation."""

    class MockEvaluator:
        def __init__(self, should_pass: bool):
            self.should_pass = should_pass

        async def __call__(self, result: MultimodalContent) -> EvaluatorResult:
            return make_evaluator_result(
                score=0.8 if self.should_pass else 0.3,
                threshold=0.5,
            )

    # Test passing evaluation
    passing_evaluator = MockEvaluator(should_pass=True)
    eval_stage = Stage.result_evaluation(passing_evaluator)

    initial_state = StageState.of(context=(), result=MultimodalContent.of("good result"))

    result_state = await eval_stage(state=initial_state)
    assert result_state == initial_state  # Should pass through unchanged

    # Test failing evaluation
    failing_evaluator = MockEvaluator(should_pass=False)
    fail_raising_eval_stage = Stage.result_evaluation(failing_evaluator, raises=True)

    with raises(StageException) as exc_info:
        await fail_raising_eval_stage(state=initial_state)

    assert "Result evaluation failed" in str(exc_info.value)
    assert exc_info.value.meta["evaluation_performance"] == 60.0  # (0.3/0.5) * 100


@mark.asyncio
async def test_stage_state_operations():
    """Test StageState operations."""
    # Test creation
    test_data = ExampleData(value="test", count=1)
    test_state_obj = ExampleState(data="custom")

    state = StageState.of(
        test_data, test_state_obj, context=(), result=MultimodalContent.of("result")
    )

    # Test state retrieval
    retrieved_data = state.get(ExampleData)
    assert retrieved_data == test_data

    retrieved_state = state.get(ExampleState, required=True)
    assert retrieved_state == test_state_obj

    # Test missing state
    assert state.get(str) is None

    with raises(MissingState):
        state.get(str, required=True)

    # Test state update
    new_data = ExampleData(value="updated", count=2)
    updated_state = state.updated(new_data, result=MultimodalContent.of("new result"))

    assert updated_state.get(ExampleData) == new_data
    assert extract_text_content(updated_state.result) == "new result"
    # Original state unchanged
    assert state.get(ExampleData) == test_data

    # Test state merging
    other_state = StageState.of(
        ExampleData(value="other", count=3),
        context=(
            ModelInput.of(MultimodalContent.of("other input")),
            ModelOutput.of(MultimodalContent.of("other completion")),
        ),
        result=MultimodalContent.of("other result"),
    )

    merged_state = state.merged(other_state)
    assert len(merged_state.context) == 2  # Combined contexts (0 + 2)
    assert "resultother result" in extract_text_content(merged_state.result)


@mark.asyncio
async def test_stage_immutability():
    """Test that stages are immutable."""
    stage = Stage.completion("Test")

    # Should not be able to modify stage attributes
    with raises(RuntimeError):
        stage.some_attribute = "value"

    with raises(RuntimeError):
        del stage.meta


@mark.asyncio
async def test_stage_with_ctx():
    """Test stage execution with custom context."""
    custom_state = ExampleState(data="custom_context")

    async def mock_generating(**kwargs) -> ModelOutput:
        # Verify custom state is available in context
        current_state = ctx.state(ExampleState)
        assert current_state.data == "custom_context"
        return ModelOutput.of(MultimodalContent.of("Context verified"))

    async with ctx.scope("test", GenerativeModel(generating=mock_generating)):
        stage = Stage.completion("Test").with_ctx(custom_state)

        initial_state = StageState.of(context=(), result=MultimodalContent.empty)

        result_state = await stage(state=initial_state)
        assert extract_text_content(result_state.result) == "Context verified"
