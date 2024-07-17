from collections.abc import Mapping, Sequence
from typing import TypeVar

from draive import DataModel
from draive.parameters import ParametersSpecification
from draive.parameters.specification import parameter_specification


def test_specifications() -> None:
    assert parameter_specification(
        annotation=str,
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "string",
    }
    assert parameter_specification(
        annotation=int,
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "integer",
    }
    assert parameter_specification(
        annotation=float,
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "number",
    }
    assert parameter_specification(
        annotation=bool,
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "boolean",
    }
    assert parameter_specification(
        annotation=None,
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "null",
    }
    assert parameter_specification(
        annotation=list[str],
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "items": {"type": "string"}}
    assert parameter_specification(
        annotation=tuple[str, ...],
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "items": {"type": "string"}}
    assert parameter_specification(
        annotation=tuple[str, str],
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "prefixItems": [{"type": "string"}, {"type": "string"}]}
    assert parameter_specification(
        annotation=Sequence[str],
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "array", "items": {"type": "string"}}
    assert parameter_specification(
        annotation=dict[str, str],
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "object", "additionalProperties": {"type": "string"}}
    assert parameter_specification(
        annotation=Mapping[str, str],
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"type": "object", "additionalProperties": {"type": "string"}}
    assert parameter_specification(
        annotation=str | int,
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert parameter_specification(
        annotation=TypeVar("Parameter", bound=int),
        description=None,
        type_arguments={},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "integer",
    }
    assert parameter_specification(
        annotation=TypeVar("Parameter"),
        description=None,
        type_arguments={"Parameter": int},
        globalns=globals(),
        localns=None,
        recursion_guard=frozenset(),
    ) == {
        "type": "integer",
    }


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


def test_parametrized_specification() -> None:
    class TestModel[Param](DataModel):
        param: Param

    assert TestModel.__PARAMETERS_SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    {"type": "array"},
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                ],
            },
        },
        "required": ["param"],
    }
    assert TestModel[str].__PARAMETERS_SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {"type": "string"},
        },
        "required": ["param"],
    }
    assert TestModel[int].__PARAMETERS_SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {"type": "integer"},
        },
        "required": ["param"],
    }
    assert TestModel[str] == TestModel[str]
    assert TestModel[int] == TestModel[int]
    assert TestModel[str] != TestModel[int]


def test_nested_parametrized_specification() -> None:
    class TestModelNested[Param](DataModel):
        param: Param

    class TestModelHolder[Param: DataModel](DataModel):
        param: Param

    class TestModel[Param: DataModel](DataModel):
        param: TestModelHolder[Param]

    assert TestModel.__PARAMETERS_SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["param"],
            },
        },
        "required": ["param"],
    }
    assert TestModel[TestModelNested[str]].__PARAMETERS_SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "object",
                        "properties": {
                            "param": {
                                "type": "string",
                            },
                        },
                        "required": ["param"],
                    }
                },
                "required": ["param"],
            }
        },
        "required": ["param"],
    }
    assert TestModel[TestModelNested[int]].__PARAMETERS_SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "object",
                        "properties": {
                            "param": {
                                "type": "integer",
                            },
                        },
                        "required": ["param"],
                    }
                },
                "required": ["param"],
            }
        },
        "required": ["param"],
    }
    assert TestModel[TestModelNested[str]] == TestModel[TestModelNested[str]]
    assert TestModel[TestModelNested[int]] == TestModel[TestModelNested[int]]
    assert TestModel[TestModelNested[str]] != TestModel[TestModelNested[int]]
