import json
from typing import Any

import pytest
from haiway import TypeSpecification

from draive.utils.schema import simplified_schema


def _render(specification: dict[str, Any]) -> Any:
    return json.loads(simplified_schema({"properties": {"p": specification}}))["p"]  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("specification", "expected"),
    (
        ({"type": "null"}, "null"),
        ({"type": "null", "description": "d"}, "null(d)"),
        ({"type": "boolean"}, "boolean"),
        ({"type": "boolean", "description": "d"}, "boolean(d)"),
        ({"type": "boolean", "enum": [True]}, "true"),
        ({"type": "integer"}, "integer"),
        ({"type": "integer", "description": "d"}, "integer(d)"),
        ({"type": "integer", "enum": [1, 2]}, "1|2"),
        ({"type": "integer", "enum": [1, 2], "description": "d"}, "1|2(d)"),
        ({"type": "number", "enum": [1.5]}, "1.5"),
        ({"type": "string", "enum": ["a", "b"]}, "'a'|'b'"),
        ({"enum": ["a"]}, "'a'"),
        ({"type": ["string", "null"]}, "string|null"),
        ({"type": "object", "additionalProperties": True}, {}),
    ),
)
def test_property_rendering(specification: dict[str, Any], expected: Any) -> None:
    assert _render(specification) == expected


def test_indent_is_applied() -> None:
    specification: TypeSpecification = {"properties": {"p": {"type": "integer"}}}  # pyright: ignore[reportAssignmentType]

    assert simplified_schema(specification, indent=2) == '{\n  "p": "integer"\n}'


@pytest.mark.parametrize(
    "specification",
    (
        {"type": "weird"},
        {"properties": {"p": {"type": "weird"}}},
    ),
    ids=("root", "property"),
)
def test_unsupported_specification_raises(specification: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Unsupported basic specification"):
        simplified_schema(specification)  # pyright: ignore[reportArgumentType]


def test_simplified_schema_renders_mapping_of_values() -> None:
    specification: TypeSpecification = {
        "type": "object",
        "properties": {
            "labels": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": ["labels"],
        "additionalProperties": False,
    }

    assert json.loads(simplified_schema(specification)) == {"labels": {"<key>": "string"}}
