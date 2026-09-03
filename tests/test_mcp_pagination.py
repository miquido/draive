from collections.abc import Mapping
from typing import Any, cast

import pytest
from mcp.types import ListResourcesResult
from mcp.types import Resource as MCPResource

from draive import Paginated, Pagination
from draive.mcp.client import MCPClient, MCPClients
from draive.resources import ResourceReference


class _SessionManager:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


class _DummySession:
    def __init__(
        self,
        pages: Mapping[str | None, tuple[tuple[str, ...], str | None]],
    ) -> None:
        self._pages = pages
        self.cursors: list[str | None] = []

    async def list_resources(self, params: Any = None) -> Any:
        cursor: str | None = params.cursor if params is not None else None
        self.cursors.append(cursor)
        names, next_cursor = self._pages[cursor]
        resources = [
            MCPResource(
                uri=f"mcp://source/{name}",
                name=name,
                description=f"desc-{name}",
                mime_type="text/plain",
            )
            for name in names
        ]
        return ListResourcesResult(
            resources=resources,
            next_cursor=next_cursor,
        )


class _DummyRepository:
    def __init__(self, values: Mapping[str | None, tuple[tuple[str, ...], str | None]]) -> None:
        self._values = values

    async def fetch_list(
        self,
        *,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        _ = extra
        assert pagination is not None
        items, next_token = self._values[pagination.token]
        references = tuple(
            ResourceReference.of(f"mcp://{item}", mime_type="text/plain") for item in items
        )
        return Paginated[ResourceReference].of(
            references,
            pagination=pagination.with_token(next_token),
        )


def _mcp_client_with_session(session: _DummySession) -> MCPClient:
    client = MCPClient(
        "source",
        session_manager=_SessionManager(),
        features=(),
        tags=(),
    )
    client._session = session
    return client


@pytest.mark.asyncio
async def test_mcp_resources_list_defaults_to_standard_page_size() -> None:
    session = _DummySession(
        {
            None: (("a", "b"), None),
        }
    )
    client = _mcp_client_with_session(session)

    page = await client.resources_list(None)

    assert [reference.uri for reference in page.items] == [
        "mcp://source.source/a",
        "mcp://source.source/b",
    ]
    assert page.pagination.limit == 32
    assert page.pagination.token is None


@pytest.mark.asyncio
async def test_mcp_resources_list_does_not_skip_items_with_small_limit() -> None:
    session = _DummySession(
        {
            None: (("a", "b", "c"), "cursor-2"),
            "cursor-2": (("d", "e"), None),
        }
    )
    client = _mcp_client_with_session(session)

    page_1 = await client.resources_list(Pagination.of(limit=2))
    assert [reference.uri for reference in page_1.items] == [
        "mcp://source.source/a",
        "mcp://source.source/b",
    ]
    assert page_1.pagination.token is not None

    page_2 = await client.resources_list(page_1.pagination)
    assert [reference.uri for reference in page_2.items] == [
        "mcp://source.source/c",
        "mcp://source.source/d",
    ]
    assert page_2.pagination.token is not None

    page_3 = await client.resources_list(page_2.pagination)
    assert [reference.uri for reference in page_3.items] == [
        "mcp://source.source/e",
    ]
    assert page_3.pagination.token is None
    assert session.cursors == [None, None, "cursor-2", "cursor-2"]


@pytest.mark.asyncio
async def test_mcp_clients_aggregate_listing_supports_continuation() -> None:
    clients = MCPClients.__new__(MCPClients)
    clients._clients = {}
    clients._prompts = {}
    clients._tools = {}
    clients._resources = {
        "a": _DummyRepository(
            {
                None: (("a-1",), "a-next"),
                "a-next": (("a-2",), None),
            }
        ),
        "b": _DummyRepository(
            {
                None: (("b-1",), "b-next"),
                "b-next": (("b-2",), None),
            }
        ),
    }

    page_1 = await clients.resources_list(
        mcp_server=None,
        pagination=Pagination.of(limit=2),
    )
    assert [reference.uri for reference in page_1.items] == [
        "mcp://a-1",
        "mcp://b-1",
    ]
    assert page_1.pagination.token is not None

    page_2 = await clients.resources_list(
        mcp_server=None,
        pagination=page_1.pagination,
    )
    assert [reference.uri for reference in page_2.items] == [
        "mcp://a-2",
        "mcp://b-2",
    ]
    assert page_2.pagination.token is None


class _FixedPageRepository:
    """Repository serving a fixed page size regardless of the requested limit."""

    def __init__(
        self,
        prefix: str,
        *,
        count: int,
        page_size: int,
    ) -> None:
        self._prefix = prefix
        self._count = count
        self._page_size = page_size
        self.requests: list[tuple[str | None, int]] = []

    async def fetch_list(
        self,
        *,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        _ = extra
        assert pagination is not None
        self.requests.append((cast(str | None, pagination.token), pagination.limit))
        start: int = int(cast(str, pagination.token)) if pagination.token is not None else 0
        end: int = min(start + self._page_size, self._count)
        references = tuple(
            ResourceReference.of(f"mcp://{self._prefix}/{index}", mime_type="text/plain")
            for index in range(start, end)
        )
        return Paginated[ResourceReference].of(
            references,
            pagination=pagination.with_token(str(end) if end < self._count else None),
        )


@pytest.mark.asyncio
async def test_mcp_clients_aggregate_listing_does_not_drop_overflowing_items() -> None:
    clients = MCPClients.__new__(MCPClients)
    clients._clients = {}
    clients._prompts = {}
    clients._tools = {}
    clients._resources = {
        "a": _FixedPageRepository("a", count=40, page_size=40),
        "b": _FixedPageRepository("b", count=40, page_size=40),
    }

    collected: list[str] = []
    pagination: Pagination | None = Pagination.of(limit=32)
    for _ in range(16):  # bounded to fail instead of looping forever
        page = await clients.resources_list(
            mcp_server=None,
            pagination=pagination,
        )
        assert len(page.items) <= 32
        collected.extend(reference.uri for reference in page.items)
        if page.pagination.token is None:
            break

        pagination = page.pagination

    else:
        raise AssertionError("aggregate listing did not complete")

    expected: list[str] = [f"mcp://a/{index}" for index in range(40)]
    expected.extend(f"mcp://b/{index}" for index in range(40))
    # every resource stays reachable exactly once across the whole listing
    assert len(collected) == len(set(collected))
    assert sorted(collected) == sorted(expected)
