from base64 import b64decode, b64encode
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from haiway import State, ctx
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    ReadResourceRequestParams,
    ReadResourceResult,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from draive.mcp.server import MCPServer, _convert_multimodal_content  # pyright: ignore[reportPrivateUsage]
from draive.multimodal import ArtifactContent, MultimodalContent
from draive.multimodal import TextContent as DraiveTextContent
from draive.resources import ResourceContent, ResourceReference
from draive.resources.template import resource
from draive.tools import tool

_CONTEXT: Any = SimpleNamespace(lifespan_context=())


class _Artifact(State):
    value: int
    items: tuple[int, ...] = ()


@tool(description="Add two numbers.")
async def add(a: int, b: int) -> str:
    """Add two numbers."""

    return str(a + b)


@tool
async def failing(value: int) -> str:
    """Always fails."""

    raise RuntimeError("tool exploded")


@tool
async def artifact() -> MultimodalContent:
    """Return an artifact."""

    return MultimodalContent.of(ArtifactContent.of(_Artifact(value=1, items=(1, 2))))


@resource(uri_template="https://api.example.com/users/{user_id}")
async def user_resource(user_id: str) -> ResourceContent:
    return ResourceContent.of(f"user:{user_id}".encode(), mime_type="text/plain")


@resource(uri_template="https://api.example.com/static")
async def static_resource() -> ResourceContent:
    return ResourceContent.of(b'{"a":1}', mime_type="application/json")


@resource(uri_template="https://api.example.com/image")
async def image_resource() -> ResourceContent:
    return ResourceContent.of(b"\x89PNG", mime_type="image/png")


_RESOURCES: list[Any] = [user_resource, static_resource, image_resource]


def _handler(server: MCPServer, method: str) -> Callable[..., Awaitable[Any]]:
    return server._server._request_handlers[method].handler  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_list_tools_exposes_names_descriptions_and_json_schema() -> None:
    server = MCPServer(name="tools", tools=[add, failing])

    async with ctx.scope("test"):
        result: ListToolsResult = await _handler(server, "tools/list")(_CONTEXT, None)

    assert [tool.name for tool in result.tools] == ["add", "failing"]
    assert result.tools[0].description == "Add two numbers."
    schema = result.tools[0].input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"a", "b"}
    assert set(schema["required"]) == {"a", "b"}


@pytest.mark.asyncio
async def test_call_tool_returns_text_content() -> None:
    server = MCPServer(name="tools", tools=[add])

    async with ctx.scope("test"):
        result: CallToolResult = await _handler(server, "tools/call")(
            _CONTEXT,
            CallToolRequestParams(name="add", arguments={"a": 1, "b": 2}),
        )

    assert not result.is_error
    assert result.content == [TextContent(type="text", text="3")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    (
        ("add", {"a": "x", "b": 1}, "Validation"),
        ("missing", {}, "unknown tool"),
        ("failing", {"value": 1}, "tool exploded"),
    ),
)
async def test_call_tool_failures_are_reported_as_error_results(
    name: str,
    arguments: dict[str, Any] | None,
    message: str,
) -> None:
    server = MCPServer(name="tools", tools=[add, failing])

    async with ctx.scope("test"):
        result: CallToolResult = await _handler(server, "tools/call")(
            _CONTEXT,
            CallToolRequestParams(name=name, arguments=arguments),
        )

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert message in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_without_required_arguments_is_an_error_result() -> None:
    server = MCPServer(name="tools", tools=[add])

    async with ctx.scope("test"):
        result: CallToolResult = await _handler(server, "tools/call")(
            _CONTEXT,
            CallToolRequestParams(name="add", arguments=None),
        )

    assert result.is_error


@pytest.mark.asyncio
async def test_call_tool_artifact_is_json_encoded_embedded_resource() -> None:
    server = MCPServer(name="tools", tools=[artifact])

    async with ctx.scope("test"):
        result: CallToolResult = await _handler(server, "tools/call")(
            _CONTEXT,
            CallToolRequestParams(name="artifact", arguments={}),
        )

    assert not result.is_error
    element = result.content[0]
    assert isinstance(element, EmbeddedResource)
    assert isinstance(element.resource, BlobResourceContents)
    assert element.resource.mime_type == "application/json"
    assert element.resource.uri == f"data:application/json;base64,{element.resource.blob}"


@pytest.mark.asyncio
async def test_resources_and_templates_are_listed_separately() -> None:
    server = MCPServer(
        name="resources",
        resources=_RESOURCES[:2],
    )

    async with ctx.scope("test"):
        resources: ListResourcesResult = await _handler(server, "resources/list")(_CONTEXT, None)
        templates: ListResourceTemplatesResult = await _handler(
            server,
            "resources/templates/list",
        )(_CONTEXT, None)

    assert [element.uri for element in resources.resources] == ["https://api.example.com/static"]
    assert [element.uri_template for element in templates.resource_templates] == [
        "https://api.example.com/users/{user_id}"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("uri", "kind", "expected"),
    (
        ("https://api.example.com/users/42", TextResourceContents, "user:42"),
        ("https://api.example.com/static", TextResourceContents, '{"a":1}'),
        (
            "https://api.example.com/image",
            BlobResourceContents,
            b64encode(b"\x89PNG").decode(),
        ),
    ),
)
async def test_read_resource_converts_content(
    uri: str,
    kind: type[TextResourceContents | BlobResourceContents],
    expected: str,
) -> None:
    server = MCPServer(
        name="resources",
        resources=_RESOURCES,
    )

    async with ctx.scope("test"):
        result: ReadResourceResult = await _handler(server, "resources/read")(
            _CONTEXT,
            ReadResourceRequestParams(uri=uri),
        )

    assert len(result.contents) == 1
    content = result.contents[0]
    assert isinstance(content, kind)
    assert content.uri == uri
    assert (content.text if isinstance(content, TextResourceContents) else content.blob) == expected


@pytest.mark.asyncio
async def test_read_unknown_resource_is_rejected() -> None:
    server = MCPServer(name="resources", resources=[user_resource])

    async with ctx.scope("test"):
        with pytest.raises(ValueError, match="is not defined"):
            await _handler(server, "resources/read")(
                _CONTEXT,
                ReadResourceRequestParams(uri="https://other.example.com/x"),
            )


def test_convert_multimodal_content_maps_every_kind() -> None:
    converted = _convert_multimodal_content(
        MultimodalContent.of(
            DraiveTextContent.of("text"),
            ResourceContent.of(b"img", mime_type="image/png"),
            ResourceContent.of(b"aud", mime_type="audio/wav"),
            ResourceContent.of(b"plain", mime_type="text/plain"),
            ResourceContent.of(b"%PDF", mime_type="application/pdf"),
        )
    )

    assert isinstance(converted[0], TextContent)
    assert isinstance(converted[1], ImageContent)
    assert isinstance(converted[2], AudioContent)
    assert isinstance(converted[3], TextContent)
    assert converted[3].text == "plain"
    assert isinstance(converted[4], EmbeddedResource)


def test_convert_multimodal_content_converts_references_to_links() -> None:
    converted = list(
        _convert_multimodal_content(
            MultimodalContent.of(
                ResourceReference.of("https://example.com/a.png", mime_type="image/png")
            )
        )
    )

    assert len(converted) == 1
    link = converted[0]
    assert isinstance(link, ResourceLink)
    assert link.uri == "https://example.com/a.png"
    assert link.mime_type == "image/png"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (None, "Validation of .a failed"),
        ({"a": 1}, "Validation of .b failed"),
    ),
)
async def test_call_tool_missing_arguments_use_validation_message(
    arguments: dict[str, Any] | None,
    expected: str,
) -> None:
    server = MCPServer(name="tools", tools=[add])

    async with ctx.scope("test"):
        result: CallToolResult = await _handler(server, "tools/call")(
            _CONTEXT,
            CallToolRequestParams(name="add", arguments=arguments),
        )

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert expected in result.content[0].text
    assert "positional argument" not in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime_type", "payload"),
    (
        ("text/plain; charset=utf-8", b"hello"),
        ("text/markdown", b"# hello"),
        ("text/csv", b"a,b"),
        ("application/ld+json", b'{"a":1}'),
        ("application/json", b'{"a":1}'),
    ),
)
async def test_read_resource_text_like_mime_types_are_text(
    mime_type: str,
    payload: bytes,
) -> None:
    @resource(uri_template="https://api.example.com/doc")
    async def doc() -> ResourceContent:
        return ResourceContent.of(payload, mime_type=mime_type)

    server = MCPServer(name="resources", resources=[doc])

    async with ctx.scope("test"):
        result: ReadResourceResult = await _handler(server, "resources/read")(
            _CONTEXT,
            ReadResourceRequestParams(uri="https://api.example.com/doc"),
        )

    content = result.contents[0]
    assert isinstance(content, TextResourceContents)
    assert content.text == payload.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime_type", "payload", "is_text"),
    (
        ("text/plain", b"\xff\xfe\x00bad", False),
        ("text/plain; charset=utf-8", b"\xff\xfe", False),
        ("text/plain; charset=ascii", "zażółć".encode(), False),
        ("text/plain; charset=unknown-charset", b"\xff\xfe", False),
        ("text/plain; charset=unknown-charset", b"hello", True),
    ),
)
async def test_read_resource_undecodable_text_is_served_as_blob(
    mime_type: str,
    payload: bytes,
    is_text: bool,
) -> None:
    @resource(uri_template="https://api.example.com/doc")
    async def doc() -> ResourceContent:
        return ResourceContent.of(payload, mime_type=mime_type)

    server = MCPServer(name="resources", resources=[doc])

    async with ctx.scope("test"):
        result: ReadResourceResult = await _handler(server, "resources/read")(
            _CONTEXT,
            ReadResourceRequestParams(uri="https://api.example.com/doc"),
        )

    content = result.contents[0]
    if is_text:
        assert isinstance(content, TextResourceContents)
        assert content.text == payload.decode()

    else:
        assert isinstance(content, BlobResourceContents)
        assert b64decode(content.blob) == payload


@pytest.mark.parametrize(
    ("mime_type", "payload"),
    (
        ("text/plain", b"\xff\xfe"),
        ("text/plain; charset=unknown-charset", b"\xff\xfe"),
    ),
)
def test_convert_multimodal_content_undecodable_text_is_embedded_blob(
    mime_type: str,
    payload: bytes,
) -> None:
    converted = _convert_multimodal_content(
        MultimodalContent.of(ResourceContent.of(payload, mime_type=mime_type))
    )

    assert isinstance(converted[0], EmbeddedResource)
    assert isinstance(converted[0].resource, BlobResourceContents)
    assert b64decode(converted[0].resource.blob) == payload


def test_convert_multimodal_content_text_subtypes_are_text() -> None:
    converted = _convert_multimodal_content(
        MultimodalContent.of(ResourceContent.of(b"# hi", mime_type="text/markdown"))
    )

    assert isinstance(converted[0], TextContent)
    assert converted[0].text == "# hi"


def test_server_accepts_mixed_resource_template_arities() -> None:
    # pyright strict verifies the mixed `resources` typing, runtime verifies construction
    server = MCPServer(
        name="resources",
        resources=[user_resource, static_resource, image_resource],
    )

    assert server is not None


@pytest.mark.parametrize("prepare", ["streamable", "sse"])
def test_asgi_apps_do_not_expose_tracebacks(prepare: str) -> None:
    server = MCPServer(name="tools", tools=[add])
    app = (
        server.prepare_streamable_http_asgi() if prepare == "streamable" else server.prepare_asgi()
    )

    assert app.debug is False
