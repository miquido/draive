import pytest

from draive.mcp.client import MCPClient, MCPClients  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("identifier", "server_uri", "listed_uri"),
    [
        ("source", "notes:///docs/a", "notes://source/docs/a"),
        ("source", "notes:///docs/a?page=2", "notes://source/docs/a?page=2"),
        ("source", "notes:///docs/a#section", "notes://source/docs/a#section"),
        ("source", "ui:///screen", "ui://source/screen"),
        ("source", "file:///tmp/item", "file://source/tmp/item"),
        ("source", "notes://host/docs/a", "notes://source.host/docs/a"),
        ("source", "mcp://host/details", "mcp://source.host/details"),
        ("source", "/details", "mcp://source/details"),
        ("my.server", "notes:///docs/a", "notes://my.server/docs/a"),
        ("my.server", "notes://host/docs/a", "notes://my.server.host/docs/a"),
        ("my.server", "/details", "mcp://my.server/details"),
    ],
)
def test_listed_resource_uri_is_restored_for_the_server(
    identifier: str,
    server_uri: str,
    listed_uri: str,
) -> None:
    client = MCPClient.stdio(identifier, command="true")

    assert client._with_uri_identifier(server_uri) == listed_uri  # pyright: ignore[reportPrivateUsage]
    assert client._without_uri_identifier(listed_uri) == server_uri  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("identifiers", "uri", "expected"),
    [
        (("source",), "notes://source/docs/a", "source"),
        (("source",), "mcp://source.item/details", "source"),
        (("my.server",), "notes://my.server/docs/a", "my.server"),
        (("my.server",), "mcp://my.server.host/details", "my.server"),
        (("my", "my.server"), "mcp://my.server.host/details", "my.server"),
        (("my", "my.server"), "mcp://my.other/details", "my"),
        (("source",), "mcp://other.source/details", None),
        (("source",), "relative/details", None),
    ],
)
def test_resource_uri_is_routed_to_the_longest_matching_client(
    identifiers: tuple[str, ...],
    uri: str,
    expected: str | None,
) -> None:
    clients = MCPClients(*(MCPClient.stdio(i, command="true") for i in identifiers))  # pyright: ignore[reportArgumentType]

    assert clients._client_identifier_for_uri(uri) == expected  # pyright: ignore[reportPrivateUsage]
