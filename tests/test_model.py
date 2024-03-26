import json
from typing import Any, Literal

from draive import Field, Model


def invalid(value: str) -> None:
    if value != "valid":
        raise ValueError()


class ExampleNestedModel(Model):
    string: str = "default"
    more: list[int] | None = None


class ExampleModel(Model):
    string: str = "default"
    number: int = Field(alias="alias", description="description", default=1)
    none_default: int | None = Field(default=None)
    value_default: int = Field(default=9)
    invalid: str = Field(validator=invalid, default="valid")
    nested: ExampleNestedModel = Field(alias="answer", default=ExampleNestedModel())
    full: Literal["A", "B"] | list[int] | str | bool | None = Field(
        alias="all",
        description="complex",
        default="",
        validator=lambda value: None,
    )


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
    }
    assert json.loads(ExampleModel.validated(**values_dict).as_json()) == values_dict


def test_validated_passes_with_default_values() -> None:
    ExampleModel.validated()
