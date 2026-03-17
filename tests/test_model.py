from collections.abc import AsyncIterable
from typing import Any

import pytest
from haiway import State, ctx

from draive.models import (
    GenerativeModel,
    ModelOutput,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent


class Example(State):
    value: str


def test_state_json_roundtrip() -> None:
    instance = Example(value="x")
    assert Example.from_json(instance.to_json()) == instance


def test_model_output_helpers() -> None:
    output = ModelOutput.of(
        MultimodalContent.of("hello"),
        ModelReasoning.of((ModelReasoningChunk.of(TextContent.of("think")),)),
        ModelToolRequest.of("call-1", tool="search", arguments={"q": "x"}),
    )

    assert output.content.to_str() == "hello"
    assert len(output.tool_requests) == 1
    assert output.without_tools().tool_requests == ()
    assert not any(isinstance(block, ModelReasoning) for block in output.without_reasoning().output)


@pytest.mark.asyncio
async def test_generative_model_completion_streams_chunks() -> None:
    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context,
        output,
        **extra: Any,
    ) -> AsyncIterable[TextContent]:
        _ = (instructions, tools, context, output, extra)
        yield TextContent.of("A")
        yield TextContent.of("B")

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        stream = GenerativeModel.completion(
            context=(),
            tools=ModelTools.none,
            output="text",
        )
        chunks = [chunk async for chunk in stream]

    assert MultimodalContent.of(*chunks).to_str() == "AB"
