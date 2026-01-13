from collections.abc import Mapping, Sequence
from typing import TypedDict

from haiway import State


def test_basic_specification_structure() -> None:
    class TestModel(State):
        str_value: str
        int_value: int
        list_value: Sequence[str]
        dict_value: Mapping[str, str]

    specification = TestModel.__SPECIFICATION__

    assert specification["type"] == "object"
    assert specification["properties"]["str_value"]["type"] == "string"
    assert specification["properties"]["int_value"]["type"] == "integer"
    assert specification["properties"]["list_value"]["type"] == "array"
    assert specification["properties"]["dict_value"]["type"] == "object"


def test_parametrized_specification_specializes_type() -> None:
    class GenericModel[T](State):
        value: T

    assert GenericModel.__SPECIFICATION__["properties"]["value"]["type"] == "object"
    assert GenericModel[str].__SPECIFICATION__["properties"]["value"]["type"] == "string"
    assert GenericModel[int].__SPECIFICATION__["properties"]["value"]["type"] == "integer"


class ExampleTypedDict(TypedDict):
    code: int


def test_typed_dict_property_is_object() -> None:
    class Wrapper(State):
        payload: ExampleTypedDict

    payload_spec = Wrapper.__SPECIFICATION__["properties"]["payload"]

    assert payload_spec["type"] == "object"
    assert payload_spec["properties"]["code"]["type"] == "integer"
