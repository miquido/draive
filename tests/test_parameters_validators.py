from __future__ import annotations

from typing import Annotated

import pytest
from haiway import ValidationError, Validator

from draive import DataModel
from draive.parameters import ParametrizedFunction


def _strict_string(value: str) -> str:
    if value != "valid":
        raise ValueError("value must be 'valid'")
    return value


def _non_negative(value: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    return value


def test_data_model_field_validator_is_executed() -> None:
    class ExampleModel(DataModel):
        value: Annotated[str, Validator(_strict_string)]

    assert ExampleModel(value="valid").value == "valid"

    with pytest.raises(ValidationError):
        ExampleModel(value="invalid")


def test_argument_validator_is_executed() -> None:
    def tool(*, value: Annotated[int, Validator(_non_negative)]) -> int:
        return value

    parametrized = ParametrizedFunction(tool)

    assert parametrized(value=1) == 1

    with pytest.raises(ValidationError):
        parametrized(value=-1)
