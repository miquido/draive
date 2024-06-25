from collections.abc import Callable
from typing import Any, Literal

from draive import MISSING, Field, Missing, State


def invalid(value: str) -> None:
    if value != "valid":
        raise ValueError()


class ExampleNestedState(State):
    string: str = "default"
    more: list[int] | None = None


def dummy_callable() -> None:
    pass


class ExampleState(State):
    string: str = "default"
    number: int = Field(aliased="alias", description="description", default=1)
    none_default: int | None = Field(default=None)
    value_default: int = Field(default=9)
    invalid: str = Field(verifier=invalid, default="valid")
    nested: ExampleNestedState = Field(aliased="answer", default=ExampleNestedState())
    full: Literal["A", "B"] | list[int] | str | bool | None = Field(
        aliased="all",
        description="complex",
        default="",
    )
    function: Callable[..., Any] = dummy_callable


# TODO: prepare extensive tests
def test_validated_passes_with_valid_values() -> None:
    values_dict: dict[str, Any] = {
        "string": "value",
        "alias": 42,
        "none_default": 1,
        "value_default": 8,
        "invalid": "valid",
        "answer": {"string": "value", "more": None},
        "all": True,
        "function": dummy_callable,
    }
    model: ExampleState = ExampleState(**values_dict)
    assert model.as_dict() == values_dict


def test_validated_passes_with_default_values() -> None:
    ExampleState()


class MissingState(State):
    value: Any | Missing


missingStateInstance: MissingState = MissingState(
    value=MISSING,
)
missingStateDict: dict[str, Any] = {"value": MISSING}


def test_missing_encoding() -> None:
    assert missingStateInstance.as_dict() == missingStateDict


def test_missing_decoding() -> None:
    assert MissingState.from_dict(missingStateDict) == missingStateInstance


class BasicsState(State):
    string: str
    string_list: list[str]
    integer: int
    integer_or_float_list: list[int | float]
    floating: float
    floating_dict: dict[str, float]
    optional: str | None
    none: None


basicStateInstance: BasicsState = BasicsState(
    string="test",
    string_list=["basic", "list"],
    integer=42,
    integer_or_float_list=[12, 3.14, 7],
    floating=9.99,
    floating_dict={"a": 65, "b": 66.0, "c": 67.5},
    optional="some",
    none=None,
)
basicStateDict: dict[str, Any] = {
    "string": "test",
    "string_list": ["basic", "list"],
    "integer": 42,
    "integer_or_float_list": [12, 3.14, 7],
    "floating": 9.99,
    "floating_dict": {"a": 65, "b": 66.0, "c": 67.5},
    "optional": "some",
    "none": None,
}


def test_basic_encoding() -> None:
    assert basicStateInstance.as_dict() == basicStateDict


def test_basic_decoding() -> None:
    assert BasicsState.from_dict(basicStateDict) == basicStateInstance


def test_type_parametrized_state() -> None:
    class Parameter(State):
        value: int

    class Parametrized[Value](State):
        value: Value

    ParametrizedAlias = Parametrized[str]
    ParametrizedParameterAlias = Parametrized[Parameter]

    # those should only instantiate without any issues
    ParametrizedAlias(value="")
    ParametrizedParameterAlias(value=Parameter(value=42))
