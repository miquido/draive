from collections.abc import Mapping, Sequence
from typing import TypedDict

from haiway import TypeSpecification

from draive import DataModel


def test_basic_specification() -> None:
    class TestModel(DataModel):
        str_value: str
        int_value: int
        float_value: float
        bool_value: bool
        none_value: None
        list_value: Sequence[str]
        dict_value: Mapping[str, str]

    specification: TypeSpecification = {
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
        "additionalProperties": False,
    }
    assert TestModel.__SPECIFICATION__ == specification


def test_parametrized_specification() -> None:
    class TestModel[Param](DataModel):
        param: Param

    assert TestModel.__SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["param"],
        "additionalProperties": False,
    }
    assert TestModel[str].__SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {"type": "string"},
        },
        "required": ["param"],
        "additionalProperties": False,
    }
    assert TestModel[int].__SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "param": {"type": "integer"},
        },
        "required": ["param"],
        "additionalProperties": False,
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

    assert TestModel.__SPECIFICATION__ == {
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
                "additionalProperties": False,
            },
        },
        "required": ["param"],
        "additionalProperties": False,
    }
    assert TestModel[TestModelNested[str]].__SPECIFICATION__ == {
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
                        "additionalProperties": False,
                    }
                },
                "required": ["param"],
                "additionalProperties": False,
            }
        },
        "required": ["param"],
        "additionalProperties": False,
    }
    assert TestModel[TestModelNested[int]].__SPECIFICATION__ == {
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
                        "additionalProperties": False,
                    }
                },
                "required": ["param"],
                "additionalProperties": False,
            }
        },
        "required": ["param"],
        "additionalProperties": False,
    }
    assert TestModel[TestModelNested[str]] == TestModel[TestModelNested[str]]
    assert TestModel[TestModelNested[int]] == TestModel[TestModelNested[int]]
    assert TestModel[TestModelNested[str]] != TestModel[TestModelNested[int]]


def test_recursive_typed_dict_specification() -> None:
    class NodeDict(TypedDict):
        value: int
        next: "NodeDict | None"

    class Wrapper(DataModel):
        node: NodeDict

    identifier = f"#{NodeDict.__qualname__}"
    assert Wrapper.__SPECIFICATION__ == {
        "type": "object",
        "properties": {
            "node": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "next": {
                        "oneOf": [
                            {"$ref": identifier},
                            {"type": "null"},
                        ],
                    },
                },
                "additionalProperties": False,
                "required": ["value", "next"],
                "$id": identifier,
            },
        },
        "required": ["node"],
        "additionalProperties": False,
    }


def test_recursive_typed_dict_references_use_identifier() -> None:
    class NodeDict(TypedDict):
        value: int
        next: "NodeDict | None"
        sibling: "NodeDict | None"

    class Wrapper(DataModel):
        node: NodeDict

    node_spec = Wrapper.__SPECIFICATION__["properties"]["node"]
    identifier = node_spec["$id"]

    assert identifier.startswith("#")
    for relation in ("next", "sibling"):
        alternatives = node_spec["properties"][relation]["oneOf"]
        assert {"$ref": identifier} in alternatives
        assert {"type": "null"} in alternatives


def test_non_recursive_typed_dict_has_no_identifier() -> None:
    class SimpleDict(TypedDict):
        value: int

    class Wrapper(DataModel):
        payload: SimpleDict

    payload_spec = Wrapper.__SPECIFICATION__["properties"]["payload"]
    assert "$id" not in payload_spec
