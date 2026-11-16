from typing import Any, cast

import pytest
from mcp.types import ListResourcesResult, ReadResourceResult, TextResourceContents
from mcp.types import Resource as MCPResource

from draive.mcp.client import MCPClient, MCPClients
from draive.resources import ResourceContent, ResourcesRepository


class _StubSession:
    """MCP session serving a fixed set of resources and recording every read."""

    def __init__(
        self,
        uris: tuple[str, ...],
    ) -> None:
        self.uris: tuple[str, ...] = uris
        self.requested: list[str] = []

    async def initialize(self) -> None:
        return None

    async def list_resources(self, params: Any = None) -> ListResourcesResult:
        _ = params
        return ListResourcesResult(
            resources=[
                MCPResource(
                    uri=uri,
                    name=uri,
                    mime_type="text/plain",
                )
                for uri in self.uris
            ],
            next_cursor=None,
        )

    async def read_resource(self, uri: str) -> ReadResourceResult:
        self.requested.append(uri)
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=uri,
                    text="payload",
                    mime_type="text/plain",
                )
            ]
        )


class _StubSessionManager:
    def __init__(
        self,
        session: _StubSession,
    ) -> None:
        self.session: _StubSession = session

    async def __aenter__(self) -> _StubSession:
        return self.session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


def _client(
    identifier: str,
    session: _StubSession,
) -> MCPClient:
    return MCPClient(
        identifier,
        session_manager=cast(Any, _StubSessionManager(session)),
        features=(ResourcesRepository,),
        tags=(),
    )


def _repository(
    features: Any,
) -> ResourcesRepository:
    return next(state for state in features if isinstance(state, ResourcesRepository))


@pytest.mark.parametrize(
    "uri",
    [
        "notes:///docs/a",
        "notes:///docs/a?page=2",
        "notes:///docs/a#section",
        "notes:/docs/a",
        "notes:/docs/a?page=2",
        "notes:/docs/a#section",
        "ui:///screen",
        "mcp://host/details",
        "file:///tmp/item",
    ],
)
@pytest.mark.asyncio
async def test_a_listed_resource_is_fetched_by_the_uri_the_server_published(uri: str) -> None:
    # `resources/list` prefixes every uri, `resources/read` has to undo it exactly -
    # a scheme without an authority (`scheme:///path`) lost a slash on the way back
    session = _StubSession((uri,))
    client = _client("source", session)

    try:
        repository = _repository(await client.__aenter__())
        listed = await repository.fetch_list()

        assert [reference.uri for reference in listed.items] != [uri]
        content = await repository.fetch(listed.items[0])

        assert isinstance(content.resource, ResourceContent)
        assert content.resource.to_bytes() == b"payload"
        assert session.requested == [uri]

    finally:
        await client.__aexit__(None, None, None)


@pytest.mark.parametrize(
    ("identifier", "uri"),
    [
        ("my.server", "mcp://host/details"),
        ("my.server", "notes:///docs/a"),
        ("mcp.source.v1", "mcp://host:8080/details"),
        ("source", "mcp://host/details"),
    ],
)
@pytest.mark.asyncio
async def test_a_client_identifier_with_dots_is_stripped_from_a_listed_uri(
    identifier: str,
    uri: str,
) -> None:
    # an identifier is an arbitrary string, a dotted one is not a scoped uri prefix
    session = _StubSession((uri,))
    client = _client(identifier, session)

    try:
        repository = _repository(await client.__aenter__())
        listed = await repository.fetch_list()
        await repository.fetch(listed.items[0])

        assert session.requested == [uri]

    finally:
        await client.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_aggregated_resources_route_to_a_client_with_a_dotted_identifier() -> None:
    session = _StubSession(("mcp://host/details",))
    clients = MCPClients(_client("my.server", session))

    try:
        await clients.__aenter__()
        listed = await clients.resources_list(pagination=None)

        assert [reference.uri for reference in listed.items] == [
            "mcp://my.server.host/details",
        ]
        fetched = await clients.resource_fetch(listed.items[0].uri)

        assert fetched is not None
        assert session.requested == ["mcp://host/details"]

    finally:
        await clients.__aexit__(None, None, None)


@pytest.mark.parametrize(
    "published_uri",
    [
        "mcp://host.api/details",
        "mcp://source.host/details",
    ],
)
@pytest.mark.asyncio
async def test_aggregated_resources_keep_originating_client_when_identifiers_overlap(
    published_uri: str,
) -> None:
    source_session = _StubSession((published_uri,))
    nested_session = _StubSession(())
    clients = MCPClients(
        _client("source", source_session),
        _client("source.host", nested_session),
    )

    try:
        await clients.__aenter__()
        listed = await clients.resources_list(pagination=None)

        assert len(listed.items) == 1
        fetched = await clients.resource_fetch(listed.items[0].uri)

        assert fetched is not None
        assert source_session.requested == [published_uri]
        assert nested_session.requested == []

    finally:
        await clients.__aexit__(None, None, None)
