from collections.abc import Callable
from typing import Any, Literal

from draive import Field, State


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
    number: int = Field(alias="alias", description="description", default=1)
    none_default: int | None = Field(default=None)
    value_default: int = Field(default=9)
    invalid: str = Field(validator=invalid, default="valid")
    nested: ExampleNestedState = Field(alias="answer", default=ExampleNestedState())
    full: Literal["A", "B"] | list[int] | str | bool | None = Field(
        alias="all",
        description="complex",
        default="",
        validator=lambda value: None,
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
    model: ExampleState = ExampleState.validated(**values_dict)
    assert model.as_dict() == values_dict


def test_validated_passes_with_default_values() -> None:
    ExampleState.validated()
