import random

import pytest
from haiway import Meta

from draive import MultimodalContent, TextContent
from draive.models import (
    ModelOutputChunk,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    model_output_blocks,
)
from draive.multimodal import MultimodalTag
from draive.resources import ResourceReference

_A = Meta.of({"k": "a"})
_B = Meta.of({"k": "b"})


def _naive(elements: list[TextContent | str | ResourceReference]) -> list[tuple[str, Meta]]:
    # reference semantics: adjacent texts merge when the meta is identical, plain strings
    # merge into texts without meta
    merged: list[tuple[str, Meta] | ResourceReference] = []
    for element in elements:
        if isinstance(element, ResourceReference):
            merged.append(element)
            continue

        text: str
        meta: Meta
        if isinstance(element, str):
            text, meta = element, Meta.empty

        else:
            text, meta = element.text, element.meta

        if merged and not isinstance(merged[-1], ResourceReference) and merged[-1][1] == meta:
            merged[-1] = (merged[-1][0] + text, meta)

        else:
            merged.append((text, meta))

    return [item for item in merged if not isinstance(item, ResourceReference)]


@pytest.mark.parametrize("seed", range(20))
def test_content_merging_matches_pairwise_semantics(seed: int) -> None:
    rng = random.Random(seed)
    reference = ResourceReference.of("https://example.com/a.png", mime_type="image/png")
    elements: list[TextContent | str | ResourceReference] = []
    for index in range(rng.randint(1, 60)):
        match rng.randint(0, 4):
            case 0:
                elements.append(f"s{index}")

            case 1:
                elements.append(TextContent.of(f"t{index}"))

            case 2:
                elements.append(TextContent.of(f"a{index}", meta=_A))

            case 3:
                elements.append(TextContent.of(f"b{index}", meta=_B))

            case _:
                elements.append(reference)

    content = MultimodalContent.of(*elements)

    # texts between references are compared with the reference semantics
    expected: list[tuple[str, Meta]] = []
    segment: list[TextContent | str | ResourceReference] = []
    for element in [*elements, None]:
        if element is None or isinstance(element, ResourceReference):
            expected.extend(_naive(segment))
            segment = []

        else:
            segment.append(element)

    assert [
        (part.text, part.meta) for part in content.parts if isinstance(part, TextContent)
    ] == expected
    assert sum(isinstance(part, ResourceReference) for part in content.parts) == sum(
        isinstance(element, ResourceReference) for element in elements
    )


def test_merging_keeps_meta_of_the_first_text_of_a_run() -> None:
    content = MultimodalContent.of(
        TextContent.of("a", meta=_A),
        TextContent.of("b", meta=_A),
        TextContent.of("c", meta=_B),
        TextContent.of("d", meta=_B),
    )

    assert [(part.text, part.meta) for part in content.texts()] == [("ab", _A), ("cd", _B)]


def test_merging_content_and_tag_parts_into_the_same_run() -> None:
    content = MultimodalContent.of(
        "x",
        MultimodalContent.of("y", "z"),
        MultimodalTag.of("w", name="t"),
    )

    assert content.to_str().replace("\n", "") == "xyz<t>w</t>"


def test_merging_many_chunks_is_exact() -> None:
    chunks = [TextContent.of(f"{index},") for index in range(5000)]

    content = MultimodalContent.of(*chunks)

    assert len(content.parts) == 1
    assert content.to_str() == "".join(chunk.text for chunk in chunks)


def test_output_blocks_of_long_streams_keep_order_and_grouping() -> None:
    reasoning = [
        ModelReasoningChunk.of(TextContent.of(f"r{index}"), final=index == 2) for index in range(3)
    ]
    request = ModelToolRequest.of("id", tool="lookup", arguments={"a": 1})
    chunks: list[ModelOutputChunk] = [
        *[TextContent.of(f"a{index}") for index in range(3000)],
        *reasoning,
        *[TextContent.of(f"b{index}") for index in range(3000)],
        request,
        TextContent.of("tail"),
    ]

    blocks = model_output_blocks(chunks)

    assert len(blocks) == 5
    first, second, third, fourth, fifth = blocks
    assert isinstance(first, MultimodalContent)
    assert first.to_str() == "".join(f"a{index}" for index in range(3000))
    assert isinstance(second, ModelReasoning)
    assert isinstance(third, MultimodalContent)
    assert third.to_str() == "".join(f"b{index}" for index in range(3000))
    assert fourth == request
    assert isinstance(fifth, MultimodalContent)
    assert fifth.to_str() == "tail"


def test_output_blocks_group_non_text_parts_with_text_between_reasoning() -> None:
    reference = ResourceReference.of("https://example.com/a.png", mime_type="image/png")
    chunks: list[ModelOutputChunk] = [
        TextContent.of("one"),
        reference,
        TextContent.of("two"),
        ModelReasoningChunk.of(TextContent.of("think"), final=True),
        TextContent.of("three"),
    ]

    blocks = model_output_blocks(chunks)

    assert len(blocks) == 3
    assert isinstance(blocks[0], MultimodalContent)
    assert [type(part) for part in blocks[0].parts] == [TextContent, ResourceReference, TextContent]
    assert isinstance(blocks[1], ModelReasoning)
    assert isinstance(blocks[2], MultimodalContent)
    assert blocks[2].to_str() == "three"
