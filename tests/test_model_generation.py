from collections.abc import Iterable
from typing import Any

import pytest
from haiway import State, ctx

from draive.generation.model import ModelGeneration
from draive.multimodal import MultimodalContent
from draive.tools import tool


class Person(State):
    name: str


@pytest.mark.asyncio
async def test_model_generation_uses_custom_generator() -> None:
    captured: dict[str, Any] = {}

    async def generating(
        generated: type[State],
        /,
        *,
        instructions: str,
        toolbox,
        examples: Iterable[tuple[str, State]],
        decoder,
        **extra: Any,
    ) -> State:
        input_text = extra.pop("input")
        assert isinstance(input_text, str)
        captured["generated"] = generated
        captured["instructions"] = instructions
        captured["input"] = input_text
        captured["tool_names"] = tuple(toolbox.tools.keys())
        captured["examples"] = tuple(examples)
        captured["decoder"] = decoder
        captured["extra"] = extra
        return Person(name="Ada")

    @tool
    async def ping() -> str:
        return "pong"

    async with ctx.scope("test", ModelGeneration(generating=generating)):
        result = await ModelGeneration.generate(
            Person,
            instructions="return a person",
            input="any",
            tools=[ping],
            trace_id="trace-1",
        )

    assert result == Person(name="Ada")
    assert captured["generated"] is Person
    assert captured["instructions"] == "return a person"
    assert captured["input"] == "any"
    assert captured["tool_names"] == ("ping",)
    assert captured["examples"] == ()
    assert captured["decoder"] is None
    assert captured["extra"]["trace_id"] == "trace-1"


@pytest.mark.asyncio
async def test_model_generation_passes_decoder_to_generator() -> None:
    async def generating(
        generated: type[State],
        /,
        *,
        instructions: str,
        toolbox,
        examples,
        decoder,
        **extra: Any,
    ) -> State:
        input_text = extra.pop("input")
        assert isinstance(input_text, str)
        _ = (generated, instructions, input_text, toolbox, examples, extra)
        assert decoder is not None
        return decoder(MultimodalContent.of('{"name":"Grace"}'))

    def decoder(content: MultimodalContent) -> Person:
        return Person.from_json(content.to_str())

    async with ctx.scope("test", ModelGeneration(generating=generating)):
        result = await ModelGeneration.generate(
            Person,
            instructions="decode",
            input="payload",
            decoder=decoder,
        )

    assert result == Person(name="Grace")
