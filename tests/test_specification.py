from collections.abc import Mapping, Sequence

from draive import DataModel
from draive.parameters import ParametersSpecification
from draive.parameters.specification import parameter_specification


def test_specifications() -> None:
    assert parameter_specification(
        annotation=str,
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "string",
    }
    assert parameter_specification(
        annotation=int,
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "integer",
    }
    assert parameter_specification(
        annotation=float,
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "number",
    }
    assert parameter_specification(
        annotation=bool,
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "boolean",
    }
    assert parameter_specification(
        annotation=None,
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "null",
    }
    assert parameter_specification(
        annotation=list[str],
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "items": {"type": "string"}}
    assert parameter_specification(
        annotation=tuple[str, ...],
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "items": {"type": "string"}}
    assert parameter_specification(
        annotation=tuple[str, str],
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "prefixItems": [{"type": "string"}, {"type": "string"}]}
    assert parameter_specification(
        annotation=Sequence[str],
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "items": {"type": "string"}}
    assert parameter_specification(
        annotation=dict[str, str],
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "object", "additionalProperties": {"type": "string"}}
    assert parameter_specification(
        annotation=Mapping[str, str],
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "object", "additionalProperties": {"type": "string"}}
    assert parameter_specification(
        annotation=str | int,
        description=None,
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}


def test_basic_specification() -> None:
    class TestModel(DataModel):
        str_value: str
        int_value: int
        float_value: float
        bool_value: bool
        none_value: None
        list_value: list[str]
        dict_value: dict[str, str]

    specification: ParametersSpecification = {
        "type": "object",
        "properties": {
            "str_value": {"type": "string"},
            "int_value": {"type": "integer"},
            "float_value": {"type": "number"},
            "bool_value": {"type": "boolean"},
            "none_value": {"type": "null"},
            "list_value": {"type": "array", "items": {"type": "string"}},
            "dict_value": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        "required": [
            "str_value",
            "int_value",
            "float_value",
            "bool_value",
            "none_value",
            "list_value",
            "dict_value",
        ],
    }
    assert TestModel.__PARAMETERS_SPECIFICATION__ == specification
