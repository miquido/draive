from collections.abc import Sequence
from typing import Annotated, Any, Literal

from haiway import Alias, Description, State

from draive.utils.schema import simplified_schema


class SchemaNestedModel(State):
    value: Annotated[Literal["A", "B", "C"], Description("selection")]
    other_value: None


class SchemaModel(State):
    str_value: str
    int_value: int
    float_value: float
    bool_value: bool
    list_value: Sequence[str]
    optional_value: Annotated[str | None, Description("alternative")]
    nested: Annotated[SchemaNestedModel, Alias("nested_value"), Description("alternative")]


def test_json_schema_contains_expected_keys() -> None:
    schema = SchemaModel.json_schema(indent=2)

    assert '"type": "object"' in schema
    assert '"nested_value"' in schema
    assert '"optional_value"' in schema


def test_simplified_schema_contains_expected_markers() -> None:
    summary = simplified_schema(SchemaModel.__SPECIFICATION__, indent=2)

    assert "str_value" in summary
    assert "nested_value" in summary
    assert "'A'|'B'|'C'" in summary


class SchemaRecursiveModel(State):
    name: str
    children: Sequence[SchemaRecursiveModel]


def test_simplified_schema_renders_recursive_reference() -> None:
    summary = simplified_schema(SchemaRecursiveModel.__SPECIFICATION__)

    assert '"children": ["#SchemaRecursiveModel"]' in summary


class SchemaAlternativesModel(State):
    mixed: Literal["a", 1, None]
    flag: Literal[True]
    union: str | Sequence[int]
    anything: Any


def test_simplified_schema_renders_alternatives() -> None:
    summary = simplified_schema(SchemaAlternativesModel.__SPECIFICATION__)

    assert "'a'|1|null" in summary
    assert '"flag": "true"' in summary
    assert '"union": "string|[\\"integer\\"]"' in summary
    assert "string|number|integer|boolean|object|array|null" in summary
