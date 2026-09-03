from mcp.types import ServerCapabilities

from draive.mcp.server import MCPServer
from draive.resources.template import resource
from draive.resources.types import ResourceContent
from draive.tools import tool


@resource(uri_template="https://api.example.com/users/{user_id}")
async def user_resource(user_id: str) -> ResourceContent:
    return ResourceContent.of(f"user:{user_id}".encode(), mime_type="text/plain")


@tool
async def add(a: int, b: int) -> str:
    """Add two numbers."""

    return str(a + b)


def _capabilities(server: MCPServer) -> ServerCapabilities:
    return server._server.get_capabilities()  # pyright: ignore[reportPrivateUsage]


def _handlers(server: MCPServer) -> set[str]:
    return {
        str(request)
        for request in server._server._request_handlers  # pyright: ignore[reportPrivateUsage]
    }


def test_templates_only_server_advertises_resources_capability() -> None:
    server = MCPServer(
        name="templates",
        resources=(user_resource,),
    )

    # the capability is derived from the `resources/list` handler registration
    assert _capabilities(server).resources is not None
    handlers: set[str] = _handlers(server)
    assert "resources/list" in handlers
    assert "resources/templates/list" in handlers
    assert "resources/read" in handlers


def test_server_without_resources_does_not_advertise_resources_capability() -> None:
    server = MCPServer(name="empty")

    assert _capabilities(server).resources is None
    assert "resources/list" not in _handlers(server)


def test_server_with_exhausted_resources_iterable_does_not_advertise_capability() -> None:
    server = MCPServer(
        name="empty",
        # a generator is truthy even when it yields nothing
        resources=(element for element in ()),
    )

    assert _capabilities(server).resources is None
    assert "resources/list" not in _handlers(server)


def test_server_without_tools_does_not_advertise_tools_capability() -> None:
    # `Toolbox.empty` is truthy, an empty toolbox must not advertise tools
    server = MCPServer(name="empty")

    assert _capabilities(server).tools is None
    assert "tools/list" not in _handlers(server)


def test_server_with_tools_advertises_tools_capability() -> None:
    server = MCPServer(
        name="tools",
        tools=(add,),
    )

    assert _capabilities(server).tools is not None
    handlers: set[str] = _handlers(server)
    assert "tools/list" in handlers
    assert "tools/call" in handlers
