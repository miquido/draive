from collections.abc import Mapping
from typing import Any

import pytest
from mcp.types import CallToolResult, ListToolsResult
from mcp.types import ResourceLink as MCPResourceLink
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool

from draive.mcp.client import MCPClient
from draive.tools import ToolException


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
        pages: Mapping[str | None, tuple[tuple[str, ...], str | None]] | None = None,
        result: CallToolResult | None = None,
    ) -> None:
        self._pages = pages if pages is not None else {}
        self._result = result
        self.cursors: list[str | None] = []

    async def list_tools(self, *, params: Any = None) -> Any:
        cursor: str | None = params.cursor if params is not None else None
        self.cursors.append(cursor)
        names, next_cursor = self._pages[cursor]
        return ListToolsResult(
            tools=[
                MCPTool(
                    name=name,
                    description=f"desc-{name}",
                    input_schema={"type": "object", "properties": {}},
                )
                for name in names
            ],
            next_cursor=next_cursor,
        )

    async def call_tool(
        self,
        name: str,
        arguments: Any = None,
    ) -> CallToolResult:
        _ = (name, arguments)
        assert self._result is not None  # nosec: B101
        return self._result


def _mcp_client_with_session(session: _DummySession) -> MCPClient:
    client = MCPClient(
        "source",
        session_manager=_SessionManager(),
        features=(),
        tags=(),
    )
    client._session = session  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    return client


@pytest.mark.asyncio
async def test_mcp_tools_fetch_follows_pagination_cursor() -> None:
    session = _DummySession(
        {
            None: (("a", "b"), "cursor-2"),
            "cursor-2": (("c",), "cursor-3"),
            "cursor-3": (("d",), None),
        }
    )
    client = _mcp_client_with_session(session)

    tools = await client.tools_fetch()

    assert [tool.name for tool in tools] == ["a", "b", "c", "d"]
    assert session.cursors == [None, "cursor-2", "cursor-3"]


@pytest.mark.asyncio
async def test_mcp_tools_fetch_stops_on_repeated_cursor() -> None:
    session = _DummySession(
        {
            None: (("a",), "cursor-2"),
            # a source repeating its cursor would loop forever otherwise
            "cursor-2": (("b",), "cursor-2"),
        }
    )
    client = _mcp_client_with_session(session)

    tools = await client.tools_fetch()

    assert [tool.name for tool in tools] == ["a", "b"]
    assert session.cursors == [None, "cursor-2"]


@pytest.mark.asyncio
async def test_mcp_tool_call_error_carries_the_reported_content() -> None:
    session = _DummySession(
        result=CallToolResult(
            content=[
                MCPTextContent(
                    type="text",
                    text="missing required argument",
                ),
                # unsupported content within an error result must not shadow the failure
                MCPResourceLink(
                    type="resource_link",
                    uri="mcp://source/details",
                    name="details",
                ),
            ],
            is_error=True,
        )
    )
    client = _mcp_client_with_session(session)

    with pytest.raises(ToolException) as error:
        await client._tool_call("failing", {})  # pyright: ignore[reportPrivateUsage]

    assert "missing required argument" in str(error.value)
    assert error.value.tool == "failing"


@pytest.mark.asyncio
async def test_mcp_tool_call_returns_converted_content() -> None:
    session = _DummySession(
        result=CallToolResult(
            content=[
                MCPTextContent(
                    type="text",
                    text="result",
                )
            ],
            is_error=False,
        )
    )
    client = _mcp_client_with_session(session)

    content = await client._tool_call("working", {})  # pyright: ignore[reportPrivateUsage]

    assert content.to_str() == "result"
