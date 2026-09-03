from typing import Any

import pytest
from jsonschema import Draft202012Validator

from draive.mcp.server import _json_schema
from draive.tools import tool


@tool
async def add(a: int, b: int) -> str:
    """Add two numbers."""

    return str(a + b)


def test_json_schema_converts_haiway_collections_to_json_types() -> None:
    schema: dict[str, Any] = _json_schema(add.specification.parameters)

    assert isinstance(schema, dict)
    # the schema is serialized with `model_dump(mode="json")`, tuples break there
    assert schema["required"] == ["a", "b"]
    assert isinstance(schema["required"], list)
    assert isinstance(schema["properties"], dict)
    assert schema["properties"]["a"] == {"type": "integer"}


def test_json_schema_is_accepted_by_metaschema_validator() -> None:
    schema: dict[str, Any] = _json_schema(add.specification.parameters)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate({"a": 1, "b": 2})


def test_json_schema_handles_missing_parameters() -> None:
    assert _json_schema(None) == {}
    assert _json_schema({}) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": ("x", "y")}, {"a": ["x", "y"]}),
        ({"a": {"b": ("x",)}}, {"a": {"b": ["x"]}}),
        ({"a": "text"}, {"a": "text"}),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_json_schema_conversion_cases(value: Any, expected: Any) -> None:
    assert _json_schema(value) == expected
