from typing import Any

import pytest

from draive.mcp.client import MCPClient, MCPClients  # pyright: ignore[reportPrivateUsage]
from draive.resources import ResourceContent, ResourcesRepository


def _repository(uri_prefix: str) -> ResourcesRepository:
    async def fetching(
        uri: str,
        **extra: Any,
    ) -> ResourceContent | None:
        _ = extra
        assert uri.startswith(uri_prefix), uri
        return ResourceContent.of(b"payload", mime_type="text/plain")

    return ResourcesRepository(fetching=fetching)


def _clients(
    *identifiers: str,
) -> MCPClients:
    clients = MCPClients.__new__(MCPClients)
    clients._clients = {  # pyright: ignore[reportPrivateUsage]
        identifier: MCPClient.stdio(identifier, command="true") for identifier in identifiers
    }
    clients._resources = {}  # pyright: ignore[reportPrivateUsage]
    return clients


@pytest.mark.asyncio
async def test_fetching_from_a_known_client_without_resources_returns_none() -> None:
    # a client may expose tools only - asking it for a resource is not an error
    clients = _clients("source")

    assert await clients.resource_fetch("mcp://source/details") is None
    assert await clients.resource_fetch("mcp://source.item/details") is None


@pytest.mark.asyncio
async def test_fetching_from_an_unknown_source_returns_none() -> None:
    clients = _clients("source")
    clients._resources = {"source": _repository("mcp://source/")}  # pyright: ignore[reportPrivateUsage]

    assert await clients.resource_fetch("mcp://unknown/details") is None
    assert await clients.resource_fetch("mcp://unknown.item/details") is None
    assert await clients.resource_fetch("mcp://other.source/details") is None
    assert await clients.resource_fetch("relative/details") is None
    assert await clients.resource_fetch("") is None


@pytest.mark.asyncio
async def test_fetching_routes_the_client_identifier() -> None:
    clients = _clients("source")
    clients._resources = {"source": _repository("mcp://source/")}  # pyright: ignore[reportPrivateUsage]

    content = await clients.resource_fetch("mcp://source/details")

    assert content is not None
    assert isinstance(content, ResourceContent)
    assert content.to_bytes() == b"payload"


@pytest.mark.asyncio
async def test_fetching_routes_a_dotted_prefix_to_its_client() -> None:
    # servers scope their resource identifiers under a dotted prefix
    clients = _clients("source")
    clients._resources = {"source": _repository("mcp://source.item/")}  # pyright: ignore[reportPrivateUsage]

    content = await clients.resource_fetch("mcp://source.item/details")

    assert content is not None
    assert isinstance(content, ResourceContent)
    assert content.to_bytes() == b"payload"


@pytest.mark.parametrize(
    ("netloc", "expected"),
    [
        ("source", "source"),
        ("source.item", "source"),
        ("source.item.nested", "source"),
        ("unknown", None),
        ("unknown.item", None),
        ("other.source", None),
        ("", None),
    ],
)
def test_client_identifier_is_resolved_only_for_known_clients(
    netloc: str,
    expected: str | None,
) -> None:
    clients = _clients("source")
    uri = f"mcp://{netloc}/details" if netloc else "relative/details"

    assert clients._client_identifier_for_uri(uri) == expected  # pyright: ignore[reportPrivateUsage]
