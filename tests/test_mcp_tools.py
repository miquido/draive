from collections.abc import Mapping
from typing import Any

import pytest
from mcp.types import CallToolResult, ListToolsResult
from mcp.types import ResourceLink as MCPResourceLink
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool

from draive import ctx
from draive.mcp.client import (  # pyright: ignore[reportPrivateUsage]
    MCPClient,
    _convert_tool,
    _tool_parameters,
    _UnsupportedSchema,
)
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


@pytest.mark.asyncio
async def test_mcp_tools_fetch_stops_on_cyclic_cursors() -> None:
    session = _DummySession(
        {
            None: (("a",), "cursor-2"),
            "cursor-2": (("b",), "cursor-3"),
            # cycling back to an already visited cursor can't advance any further
            "cursor-3": (("c",), "cursor-2"),
        }
    )
    client = _mcp_client_with_session(session)

    tools = await client.tools_fetch()

    assert [tool.name for tool in tools] == ["a", "b", "c"]
    assert session.cursors == [None, "cursor-2", "cursor-3"]


async def _unused_tool_call(name: str, arguments: Mapping[str, Any]) -> Any:
    raise AssertionError(f"unexpected call of {name} with {arguments}")


def test_mcp_tool_parameters_skip_undeclared_schema_keys() -> None:
    tool = _convert_tool(  # pyright: ignore[reportPrivateUsage]
        MCPTool(
            name="tool",
            input_schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "Arguments",
                "type": "object",
                "properties": {
                    "query": {"type": "string", "title": "Query", "minLength": 1},
                    "limit": {
                        "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
                        "default": None,
                        "title": "Limit",
                    },
                },
                "required": ["query"],
                "additionalProperties": True,
            },
        ),
        tool_call=_unused_tool_call,
        source="source",
        tags=(),
    )

    assert _plain(tool.specification.parameters) == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _converted_parameters(schema: Mapping[str, Any]) -> Any:
    return _plain(
        _convert_tool(  # pyright: ignore[reportPrivateUsage]
            MCPTool(name="tool", input_schema=schema),
            tool_call=_unused_tool_call,
            source="source",
            tags=(),
        ).specification.parameters
    )


def test_mcp_tool_parameters_inline_definitions() -> None:
    assert _converted_parameters(
        {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "title": "Items"},
                "main": {"$ref": "#/definitions/Item", "description": "Main item"},
            },
            "required": ["items"],
            "$defs": {
                "Item": {
                    "type": "object",
                    "title": "Item",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                }
            },
            "definitions": {"Item": {"type": "string"}},
        }
    ) == {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
            "main": {"type": "string", "description": "Main item"},
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def test_mcp_tool_parameters_anchor_recursive_definitions() -> None:
    assert _converted_parameters(
        {
            "type": "object",
            "properties": {
                "root": {"$ref": "#/$defs/Node"},
                "other": {"$ref": "#/$defs/Node"},
            },
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                    },
                }
            },
        }
    ) == {
        "type": "object",
        "properties": {
            "root": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#Node"}}},
                "additionalProperties": False,
                "$anchor": "Node",
            },
            "other": {"$ref": "#Node"},
        },
        "additionalProperties": False,
    }


def test_mcp_tool_parameters_translate_composition_keywords() -> None:
    assert _converted_parameters(
        {
            "type": "object",
            "properties": {
                "choice": {"oneOf": [{"const": "a"}, {"type": "integer"}]},
                "wrapped": {"allOf": [{"type": "string"}], "description": "Wrapped"},
            },
        }
    ) == {
        "type": "object",
        "properties": {
            "choice": {"anyOf": [{"enum": ["a"]}, {"type": "integer"}]},
            "wrapped": {"type": "string", "description": "Wrapped"},
        },
        "additionalProperties": False,
    }


def test_mcp_tool_parameters_merge_root_composition_arguments() -> None:
    assert _converted_parameters(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "allOf": [
                {"properties": {"b": {"type": "integer"}}, "required": ["b"]},
            ],
            "oneOf": [
                {
                    "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
                    "required": ["q", "n"],
                },
                {
                    "properties": {"id": {"type": "integer"}, "n": {"type": "integer"}},
                    "required": ["id", "n"],
                },
            ],
        }
    ) == {
        "type": "object",
        "properties": {
            "b": {"type": "integer"},
            "a": {"type": "string"},
            "q": {"type": "string"},
            "n": {"type": "integer"},
            "id": {"type": "integer"},
        },
        "required": ["b", "a", "n"],
        "additionalProperties": False,
    }


def test_mcp_tool_parameters_translate_multi_branch_all_of_and_reference_siblings() -> None:
    assert _converted_parameters(
        {
            "type": "object",
            "allOf": [
                {"properties": {"a": {"$ref": "#/$defs/A"}}, "required": ["a"]},
                {"properties": {"b": {"type": "string", "title": "B"}}},
            ],
            "properties": {"c": {"$ref": "#/$defs/A", "description": "C"}},
            "$defs": {"A": {"type": "object", "properties": {"x": {"type": "string"}}}},
        }
    ) == {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": False,
            },
            "b": {"type": "string"},
            "c": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": False,
                "description": "C",
            },
        },
        "required": ["a"],
        "additionalProperties": False,
    }


def test_mcp_tool_parameters_keep_every_alternative_of_root_branches() -> None:
    assert _converted_parameters(
        {
            "type": "object",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"mode": {"const": "search"}, "q": {"type": "string"}},
                    "required": ["mode", "q"],
                },
                {
                    "type": "object",
                    "properties": {"mode": {"const": "get"}, "id": {"type": "integer"}},
                    "required": ["mode", "id"],
                },
            ],
        }
    ) == {
        "type": "object",
        "properties": {
            "mode": {"anyOf": [{"enum": ["search"]}, {"enum": ["get"]}]},
            "q": {"type": "string"},
            "id": {"type": "integer"},
        },
        "required": ["mode"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    "schema",
    [
        # references which do not resolve to a definition
        {"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}, "$defs": {}},
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A/properties/b"}},
            "$defs": {"A": {"type": "object", "properties": {"b": {"type": "string"}}}},
        },
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/properties/b"}, "b": {"type": "string"}},
        },
        # schemas which can not be combined without changing their meaning
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/S", "type": "integer"}},
            "$defs": {"S": {"type": "string"}},
        },
        {
            "type": "object",
            "properties": {},
            "anyOf": [{"properties": {"a": {"type": "string"}}}],
            "oneOf": [{"properties": {"b": {"type": "string"}}}],
        },
        {"type": "object", "properties": {"a": {"const": "a", "enum": ["b"]}}},
        {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A", "oneOf": [{"type": "object"}]}},
            "$defs": {"A": {"type": "object", "properties": {"x": {"type": "string"}}}},
        },
        {"type": "object", "additionalProperties": {"type": "string"}},
        {"type": "string"},
        # malformed schemas
        {"type": "object", "properties": True},
        {"type": "object", "properties": {"q": {"type": "string"}}, "required": "q"},
        {"type": "object", "properties": {"x": {"allOf": [{"required": 3}], "required": ["a"]}}},
        {"type": "object", "properties": {"x": {}}},
    ],
)
def test_mcp_tool_parameters_refuse_schemas_which_can_not_be_translated(
    schema: Mapping[str, Any],
) -> None:
    with pytest.raises(_UnsupportedSchema):
        _tool_parameters("tool", schema)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_mcp_tools_fetch_skips_tools_with_unsupported_schemas() -> None:
    class Session(_DummySession):
        async def list_tools(self, *, params: Any = None) -> Any:
            _ = params
            return ListToolsResult(
                tools=[
                    MCPTool(name="any", input_schema={"type": "object", "properties": {"x": {}}}),
                    MCPTool(name="valid", input_schema={"type": "object", "properties": {}}),
                ],
                next_cursor=None,
            )

    async with ctx.scope("test"):
        tools = await _mcp_client_with_session(Session()).tools_fetch()

    assert [tool.name for tool in tools] == ["valid"]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(element) for key, element in value.items()}  # pyright: ignore[reportUnknownVariableType]

    if isinstance(value, tuple):
        return [_plain(element) for element in value]  # pyright: ignore[reportUnknownVariableType]

    return value
