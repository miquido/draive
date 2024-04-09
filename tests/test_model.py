import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

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


class DatetimeModel(Model):
    value: datetime


datetimeModelInstance: DatetimeModel = DatetimeModel(
    value=datetime.fromtimestamp(0, UTC),
)
datetimeModelJSON: str = '{"value": "1970-01-01T00:00:00+00:00"}'


def test_datetime_encoding() -> None:
    assert datetimeModelInstance.as_json() == datetimeModelJSON


def test_datetime_decoding() -> None:
    assert DatetimeModel.from_json(datetimeModelJSON) == datetimeModelInstance


class UUIDModel(Model):
    value: UUID


uuidModelInstance: UUIDModel = UUIDModel(
    value=UUID(hex="0cf728c0369348e78552e8d86d35e8b0"),
)
uuidModelJSON: str = '{"value": "0cf728c0369348e78552e8d86d35e8b0"}'


def test_uuid_encoding() -> None:
    assert uuidModelInstance.as_json() == uuidModelJSON


def test_uuid_decoding() -> None:
    assert UUIDModel.from_json(uuidModelJSON) == uuidModelInstance
