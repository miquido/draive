from typing import Any

import pytest

from draive.mistral.completions import (  # pyright: ignore[reportPrivateUsage]
    _content_chunks,
    _image_mime_type,
    _response_format,
    _tools_as_tool_config,
)
from draive.models import ModelToolSpecification
from draive.multimodal import ArtifactContent, TextContent
from draive.resources import ResourceContent, ResourceReference

_TOOL = ModelToolSpecification(name="echo", description=None, parameters=None, meta={})


def test_tool_without_parameters_gets_an_empty_object_schema() -> None:
    choice, tools = _tools_as_tool_config([_TOOL], tool_selection="auto")

    assert choice == "auto"
    assert tools[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert tools[0]["function"]["description"] == ""


@pytest.mark.parametrize(
    ("selection", "expected_choice", "expected_tools"),
    [
        ("auto", "auto", 1),
        ("required", "any", 1),
        ("none", "none", 0),
    ],
)
def test_tool_selection_maps_to_the_tool_choice(
    selection: Any,
    expected_choice: str,
    expected_tools: int,
) -> None:
    choice, tools = _tools_as_tool_config([_TOOL], tool_selection=selection)

    assert choice == expected_choice
    assert len(tools) == expected_tools


def test_named_tool_selection_forces_the_function() -> None:
    choice, tools = _tools_as_tool_config([_TOOL], tool_selection=_TOOL)

    assert choice == {"type": "function", "function": {"name": "echo"}}
    assert len(tools) == 1


def test_without_tools_the_choice_is_none_regardless_of_selection() -> None:
    assert _tools_as_tool_config(None, tool_selection="required") == ("none", [])


@pytest.mark.parametrize("output", ["auto", "text", ("text",)])
def test_text_outputs_request_no_response_format(output: Any) -> None:
    assert _response_format(output) is None


def test_json_output_requests_a_json_object() -> None:
    assert _response_format("json") == {"type": "json_object"}


@pytest.mark.parametrize("output", ["image", "audio", ("image", "audio")])
def test_non_text_outputs_are_not_supported(output: Any) -> None:
    with pytest.raises(NotImplementedError):
        _response_format(output)


def test_content_is_converted_by_kind() -> None:
    chunks = list(
        _content_chunks(
            (
                TextContent.of("text"),
                ResourceReference.of("https://example.com/a.pdf", mime_type="application/pdf"),
                ResourceReference.of("https://example.com/a.png", mime_type="image/png"),
                ResourceContent.of(b"%PDF", mime_type="application/pdf"),
                ResourceContent.of(b"plain", mime_type="text/plain"),
                ResourceContent.of(b"\x89PNG", mime_type="image/png"),
                ArtifactContent.of(TextContent.of("shown")),
                ArtifactContent.of(TextContent.of("secret"), hidden=True),
            )
        )
    )

    assert [chunk["type"] for chunk in chunks] == [
        "text",
        "document_url",
        "image_url",
        "document_url",
        "text",
        "image_url",
        "text",
    ]
    assert chunks[4]["text"] == "plain"  # pyright: ignore[reportGeneralTypeIssues]
    assert chunks[5]["image_url"]["url"].startswith("data:image/png;base64,")  # pyright: ignore


@pytest.mark.parametrize(
    "element",
    [
        ResourceReference.of("https://example.com/a.zip", mime_type="application/zip"),
        ResourceContent.of(b"zip", mime_type="application/zip"),
    ],
)
def test_unsupported_media_is_rejected(element: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        list(_content_chunks((element,)))


@pytest.mark.parametrize(
    ("url", "mime_type"),
    [
        ("https://example.com/a.jpg?x=1", "image/jpeg"),
        ("https://example.com/a.webp", "image/webp"),
        ("https://example.com/a.txt", "image/png"),
        ("https://example.com/a", "image/png"),
    ],
)
def test_image_mime_type_follows_the_extension_defaulting_to_png(
    url: str,
    mime_type: str,
) -> None:
    assert _image_mime_type(url) == mime_type


def _assert_materialized(value: Any) -> None:
    # a lazy iterable is consumed by the first serialization, a retried request would be empty
    if isinstance(value, dict):
        for item in value.values():  # pyright: ignore[reportUnknownVariableType]
            _assert_materialized(item)

    elif isinstance(value, list):
        for item in value:  # pyright: ignore[reportUnknownVariableType]
            _assert_materialized(item)

    else:
        assert isinstance(value, str | int | float | bool | type(None)), type(value)  # pyright: ignore[reportUnknownArgumentType]


def test_built_messages_are_materialized_and_serialize_identically_twice() -> None:
    import json

    from draive.mistral.completions import _build_messages  # pyright: ignore[reportPrivateUsage]
    from draive.models import (
        ModelInput,
        ModelOutput,
        ModelReasoning,
        ModelToolRequest,
        ModelToolResponse,
    )
    from draive.multimodal import MultimodalContent

    messages = _build_messages(
        instructions="system",
        context=(
            ModelInput.of(MultimodalContent.of(TextContent.of("question"))),
            ModelOutput.of(
                ModelReasoning.of(TextContent.of("thinking"), meta={"kind": "thinking"}),
                MultimodalContent.of(TextContent.of("checking")),
                ModelToolRequest.of("call-1", tool="echo", arguments={"x": 1}),
            ),
            ModelInput.of(
                MultimodalContent.of(TextContent.of("follow up")),
                ModelToolResponse.of(
                    "call-1",
                    tool="echo",
                    content=MultimodalContent.of(TextContent.of("done")),
                ),
            ),
        ),
    )
    _assert_materialized(messages)
    _, tools = _tools_as_tool_config([_TOOL], tool_selection="auto")
    _assert_materialized(tools)

    assert json.dumps(messages) == json.dumps(messages)
    assert len(json.dumps(messages)) > 100
