from draive.models import ModelReasoning, ModelToolRequest
from draive.models.generative import _merge_output
from draive.multimodal import MultimodalContent, TextContent


def test_merge_output_preserves_reasoning_and_groups_content() -> None:
    # Arrange: simulate a streaming sequence: text parts, reasoning, tool call, then text
    a = TextContent.of("a")
    b = TextContent.of("b")
    r = ModelReasoning.of(TextContent.of("r"))
    t = ModelToolRequest.of("id-1", tool="test")
    c = TextContent.of("c")

    # Act
    merged = list(_merge_output([a, b, r, t, c]))

    # Assert types and order
    assert isinstance(merged[0], MultimodalContent)
    assert isinstance(merged[1], ModelReasoning)
    assert isinstance(merged[2], ModelToolRequest)
    assert isinstance(merged[3], MultimodalContent)

    # Assert grouped content parts (text may be coalesced)
    assert merged[0].to_str() == "ab"
    assert merged[3].to_str() == "c"


def test_merge_output_merges_reasoning_with_same_meta() -> None:
    # Arrange: two reasoning chunks with same meta should be merged,
    # third with different meta separate
    r1 = ModelReasoning.of(TextContent.of("x"), meta={"id": "1"})
    r2 = ModelReasoning.of(TextContent.of("y"), meta={"id": "1"})
    t = ModelToolRequest.of("call-1", tool="t")
    r3 = ModelReasoning.of(TextContent.of("z"), meta={"id": "2"})

    # Act
    merged = list(_merge_output([r1, r2, t, r3]))

    # Assert order and merging
    assert isinstance(merged[0], ModelReasoning)
    assert isinstance(merged[1], ModelToolRequest)
    assert isinstance(merged[2], ModelReasoning)
    assert merged[0].meta.get_str("id") == "1"
    assert merged[2].meta.get_str("id") == "2"
    assert merged[0].content.to_str() == "xy"
    assert merged[2].content.to_str() == "z"
