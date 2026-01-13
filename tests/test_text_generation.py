from collections.abc import Iterable
from typing import Any

import pytest
from haiway import ctx

from draive.generation.text import TextGeneration
from draive.tools import tool


@pytest.mark.asyncio
async def test_text_generation_uses_custom_generator() -> None:
    captured: dict[str, Any] = {}

    async def generating(
        *,
        instructions: str,
        toolbox,
        examples: Iterable[tuple[str, str]],
        **extra: Any,
    ) -> str:
        input_text = extra.pop("input")
        assert isinstance(input_text, str)
        captured["instructions"] = instructions
        captured["input"] = input_text
        captured["tool_names"] = tuple(toolbox.tools.keys())
        captured["examples"] = tuple(examples)
        captured["extra"] = extra
        return "done"

    @tool
    async def ping() -> str:
        return "pong"

    async with ctx.scope("test", TextGeneration(generating=generating)):
        result = await TextGeneration.generate(
            instructions="answer shortly",
            input="hello",
            tools=[ping],
            examples=[("a", "b")],
            trace_id="trace-2",
        )

    assert result == "done"
    assert captured["instructions"] == "answer shortly"
    assert captured["input"] == "hello"
    assert captured["tool_names"] == ("ping",)
    assert captured["examples"] == (("a", "b"),)
    assert captured["extra"]["trace_id"] == "trace-2"
