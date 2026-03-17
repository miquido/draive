from collections.abc import Sequence
from typing import Annotated, Literal

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
