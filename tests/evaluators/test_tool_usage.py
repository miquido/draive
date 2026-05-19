import pytest

from draive.evaluators import ToolUsageRequirement, tool_usage_context_evaluator
from draive.models import ModelInput, ModelOutput, ModelToolRequest
from draive.multimodal import MultimodalContent


def _context(*requests: ModelToolRequest) -> tuple[ModelInput | ModelOutput, ...]:
    return (
        ModelInput.of(MultimodalContent.of("user message")),
        ModelOutput.of(*requests),
    )


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_empty_context() -> None:
    result = await tool_usage_context_evaluator((), required=["foo"])

    assert result.score == 0.0
    assert "foo" in result.meta["comment"]


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_empty_context_forbidden_only() -> None:
    result = await tool_usage_context_evaluator((), forbidden=["foo"])

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_no_requirements() -> None:
    context = _context(ModelToolRequest.of("id1", tool="foo"))

    result = await tool_usage_context_evaluator(context)

    assert result.score == 0.0
    assert result.meta["comment"] == "No tool usage requirements were provided!"


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_required_present() -> None:
    context = _context(ModelToolRequest.of("id1", tool="foo"))

    result = await tool_usage_context_evaluator(context, required=["foo"])

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_required_missing() -> None:
    context = _context(ModelToolRequest.of("id1", tool="foo"))

    result = await tool_usage_context_evaluator(context, required=["bar"])

    assert result.score == 0.0
    assert "bar" in result.meta["comment"]


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_required_with_arguments() -> None:
    context = _context(ModelToolRequest.of("id1", tool="foo", arguments={"x": 1, "y": "ok"}))

    matching = await tool_usage_context_evaluator(
        context,
        required=[ToolUsageRequirement.of("foo", arguments={"x": 1})],
    )
    assert matching.score == 1.0

    extra_ok = await tool_usage_context_evaluator(
        context,
        required=[ToolUsageRequirement.of("foo", arguments={"y": "ok"})],
    )
    assert extra_ok.score == 1.0

    mismatching = await tool_usage_context_evaluator(
        context,
        required=[ToolUsageRequirement.of("foo", arguments={"x": 2})],
    )
    assert mismatching.score == 0.0
    assert "expected arguments" in mismatching.meta["comment"]


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_expected_any() -> None:
    context = _context(ModelToolRequest.of("id1", tool="foo"))

    hit = await tool_usage_context_evaluator(context, expected=["foo", "bar"])
    assert hit.score == 1.0

    miss = await tool_usage_context_evaluator(context, expected=["alpha", "beta"])
    assert miss.score == 0.0
    assert "alpha" in miss.meta["comment"]


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_forbidden() -> None:
    context = _context(
        ModelToolRequest.of("id1", tool="foo"),
        ModelToolRequest.of("id2", tool="bar"),
    )

    clean = await tool_usage_context_evaluator(context, forbidden=["baz"])
    assert clean.score == 1.0

    dirty = await tool_usage_context_evaluator(context, forbidden=["bar"])
    assert dirty.score == 0.0
    assert "bar" in dirty.meta["comment"]


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_combined_non_strict() -> None:
    context = _context(
        ModelToolRequest.of("id1", tool="foo"),
        ModelToolRequest.of("id2", tool="bar"),
    )

    result = await tool_usage_context_evaluator(
        context,
        required=["foo", "missing"],
        forbidden=["bar"],
        strict=False,
    )

    # 1 of 3 checks pass (foo present), missing required fails, forbidden bar used
    assert result.score == pytest.approx(1 / 3)
    assert "missing" in result.meta["comment"]
    assert "bar" in result.meta["comment"]


@pytest.mark.asyncio
async def test_tool_usage_context_evaluator_no_tool_calls() -> None:
    context = (ModelInput.of(MultimodalContent.of("just text")),)

    result = await tool_usage_context_evaluator(context, required=["foo"])

    assert result.score == 0.0
