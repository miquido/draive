from typing import Any

import pytest
from haiway import State, ctx

from draive import GenerativeModel, ModelGeneration, ModelOutputChunk, TextContent
from draive.generation.model.default import _json_payload
from draive.multimodal import MultimodalContent


class Example(State, serializable=True):
    name: str


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"name": "Paris"}', '{"name": "Paris"}'),
        ('  {"name": "Paris"}  ', '{"name": "Paris"}'),
        ('```json\n{"name": "Paris"}\n```', '{"name": "Paris"}'),
        ('```\n{"name": "Paris"}\n```', '{"name": "Paris"}'),
        ('```json\n{"name": "Paris"}', '{"name": "Paris"}'),
        ("```json", "```json"),
        # trailing content past the first complete value is not a part of it
        ('{"name": "Paris"}{"name": "Kraków"}', '{"name": "Paris"}'),
        ('{"name": "Paris"}\n\nHope that helps!', '{"name": "Paris"}'),
        ('```json\n{"name": "Paris"}\n```\nAnything else?', '{"name": "Paris"}'),
        ("not json at all", "not json at all"),
    ],
)
def test_json_payload_unwraps_code_fences(raw: str, expected: str) -> None:
    assert _json_payload(MultimodalContent.of(raw)) == expected


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        # the payload can be spread across multiple parts of the completion
        (('{"name":', ' "Paris"}'), '{"name": "Paris"}'),
        (("```json\n", '{"name": "Paris"}', "\n```"), '{"name": "Paris"}'),
        (('{"name": "Paris"}', "\n\nHope that helps!"), '{"name": "Paris"}'),
        ((), ""),
    ],
)
def test_json_payload_joins_content_parts(parts: tuple[str, ...], expected: str) -> None:
    assert _json_payload(MultimodalContent.of(*parts)) == expected


@pytest.mark.asyncio
async def test_model_generation_decodes_fenced_output() -> None:
    async def _completion(**_: object):
        yield TextContent.of('```json\n{"name": "Paris"}\n```')

    async with ctx.scope("test", GenerativeModel(generating=_completion)):
        generated: Example = await ModelGeneration.generate(
            Example,
            instructions="",
            input="Describe Paris.",
        )

    assert generated.name == "Paris"


@pytest.mark.asyncio
async def test_model_generation_decoder_receives_completion_content() -> None:
    # the decoder reads back the whole completion content, not its string rendering
    received: list[MultimodalContent] = []

    async def _completion(**_: object):
        yield TextContent.of('{"name":')
        yield TextContent.of(' "Paris"}')

    def decoder(content: MultimodalContent) -> Example:
        received.append(content)
        return Example.from_json(content.to_str())

    async with ctx.scope("test", GenerativeModel(generating=_completion)):
        generated: Example = await ModelGeneration.generate(
            Example,
            instructions="",
            input="Describe Paris.",
            decoder=decoder,
        )

    assert generated.name == "Paris"
    assert len(received) == 1
    assert isinstance(received[0], MultimodalContent)
    assert received[0].to_str() == '{"name": "Paris"}'


def test_model_output_chunk_type_is_exported() -> None:
    assert ModelOutputChunk is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("decoder", (None, lambda content: Example.from_json(content.to_str())))
async def test_model_generation_always_passes_the_type_as_output(decoder: Any) -> None:
    # the type reaches the provider so it can use its schema backed mode and
    # fall back on its own - a decoder only changes how the result is read back
    captured: dict[str, Any] = {}

    async def _completion(**kwargs: Any):
        captured.update(kwargs)
        yield TextContent.of('{"name": "Paris"}')

    async with ctx.scope("test", GenerativeModel(generating=_completion)):
        generated: Example = await ModelGeneration.generate(
            Example,
            instructions="",
            input="Describe Paris.",
            decoder=decoder,
        )

    assert generated.name == "Paris"
    assert captured["output"] is Example


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("injection", "expected"),
    (
        ("skip", "Describe the city. {model_schema}"),
        ("full", '"name"'),
        ("simplified", '"name"'),
    ),
)
async def test_schema_injection_extends_the_instructions(
    injection: Any,
    expected: str,
) -> None:
    # the requested schema reaches providers without a schema backed mode through
    # the instructions, while the type is still passed along for those that have one
    captured: dict[str, Any] = {}

    async def _completion(**kwargs: Any):
        captured.update(kwargs)
        yield TextContent.of('{"name": "Paris"}')

    async with ctx.scope("test", GenerativeModel(generating=_completion)):
        await ModelGeneration.generate(
            Example,
            instructions="Describe the city. {model_schema}",
            input="Describe Paris.",
            schema_injection=injection,
        )

    assert expected in captured["instructions"]
    assert captured["output"] is Example


@pytest.mark.asyncio
async def test_model_generation_ignores_trailing_output() -> None:
    async def _completion(**_: object):
        yield TextContent.of('{"name": "Paris"}{"name": "Kraków"}')

    async with ctx.scope("test", GenerativeModel(generating=_completion)):
        generated: Example = await ModelGeneration.generate(
            Example,
            instructions="",
            input="Describe Paris.",
        )

    assert generated.name == "Paris"
