from collections.abc import AsyncIterable, Sequence
from typing import Any

import pytest
from haiway import ctx

from draive.helpers import prepare_instructions
from draive.models import GenerativeModel, ModelContextElement, ModelOutputChunk, ModelTools
from draive.multimodal import TextContent


async def _reply(text: str) -> AsyncIterable[ModelOutputChunk]:
    yield TextContent.of(text)


def _capturing_model(captured: dict[str, Any]) -> GenerativeModel:
    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        captured["instructions"] = instructions
        return _reply("<RESULT_INSTRUCTION>ok</RESULT_INSTRUCTION>")

    return GenerativeModel(generating=generating)


@pytest.mark.asyncio
async def test_guidelines_reach_the_model_verbatim() -> None:
    captured: dict[str, Any] = {}

    async with ctx.scope("test", _capturing_model(captured)):
        await prepare_instructions("summarize", guidelines="keep {topic} as it is")

    assert "<GUIDELINES>keep {topic} as it is</GUIDELINES>" in captured["instructions"]


@pytest.mark.asyncio
async def test_model_is_taught_the_template_variable_syntax() -> None:
    captured: dict[str, Any] = {}

    async with ctx.scope("test", _capturing_model(captured)):
        await prepare_instructions("summarize")

    # templates resolve `{%name%}` placeholders and the instructions are sent as they are
    assert "{%variable%}" in captured["instructions"]
    assert "{{" not in captured["instructions"]
