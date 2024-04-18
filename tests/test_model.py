import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from draive import (
    MISSING,
    AudioURLContent,
    Field,
    ImageURLContent,
    LMMCompletionMessage,
    Missing,
    Model,
)


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
    invalid: str = Field(verifier=invalid, default="valid")
    nested: ExampleNestedModel = Field(alias="answer", default=ExampleNestedModel())
    full: Literal["A", "B"] | list[int] | str | bool | None = Field(
        alias="all",
        description="complex",
        default="",
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


datetime_model_instance: DatetimeModel = DatetimeModel(
    value=datetime.fromtimestamp(0, UTC),
)
datetime_model_json: str = '{"value": "1970-01-01T00:00:00+00:00"}'


def test_datetime_encoding() -> None:
    assert datetime_model_instance.as_json() == datetime_model_json


def test_datetime_decoding() -> None:
    assert DatetimeModel.from_json(datetime_model_json) == datetime_model_instance


class UUIDModel(Model):
    value: UUID


uuid_model_instance: UUIDModel = UUIDModel(
    value=UUID(hex="0cf728c0369348e78552e8d86d35e8b0"),
)
uuid_model_json: str = '{"value": "0cf728c0369348e78552e8d86d35e8b0"}'


def test_uuid_encoding() -> None:
    assert uuid_model_instance.as_json() == uuid_model_json


def test_uuid_decoding() -> None:
    assert UUIDModel.from_json(uuid_model_json) == uuid_model_instance


class MissingModel(Model):
    value: Missing = MISSING


missing_model_instance: MissingModel = MissingModel(
    value=MISSING,
)
missing_model_json: str = '{"value": null}'


def test_missing_encoding() -> None:
    assert missing_model_instance.as_json() == missing_model_json


def test_missing_decoding() -> None:
    assert MissingModel.from_json(missing_model_json) == missing_model_instance


class BasicsModel(Model):
    string: str
    string_list: list[str]
    integer: int
    integer_or_float_list: list[int | float]
    floating: float
    floating_dict: dict[str, float]
    optional: str | None
    none: None


basic_model_instance: BasicsModel = BasicsModel(
    string="test",
    string_list=["basic", "list"],
    integer=42,
    integer_or_float_list=[12, 3.14, 7],
    floating=9.99,
    floating_dict={"a": 65, "b": 66.0, "c": 67.5},
    optional="some",
    none=None,
)
basic_model_json: str = (
    '{"string": "test",'
    ' "string_list": ["basic", "list"],'
    ' "integer": 42,'
    ' "integer_or_float_list": [12, 3.14, 7],'
    ' "floating": 9.99,'
    ' "floating_dict": {"a": 65, "b": 66.0, "c": 67.5},'
    ' "optional": "some",'
    ' "none": null}'
)


def test_basic_encoding() -> None:
    assert basic_model_instance.as_json() == basic_model_json


def test_basic_decoding() -> None:
    assert BasicsModel.from_json(basic_model_json) == basic_model_instance


basic_lmm_message_instance: LMMCompletionMessage = LMMCompletionMessage(
    role="assistant",
    content="string",
)
basic_lmm_message_json: str = '{"role": "assistant", "content": "string"}'

image_lmm_message_instance: LMMCompletionMessage = LMMCompletionMessage(
    role="assistant",
    content=ImageURLContent(image_url="https://miquido.com/image"),
)
image_lmm_message_json: str = (
    '{"role": "assistant",'
    ' "content": {"image_url": "https://miquido.com/image", "image_description": null}'
    "}"
)
audio_lmm_message_instance: LMMCompletionMessage = LMMCompletionMessage(
    role="assistant",
    content=AudioURLContent(audio_url="https://miquido.com/audio"),
)
audio_lmm_message_json: str = (
    '{"role": "assistant",'
    ' "content": {"audio_url": "https://miquido.com/audio", "audio_transcription": null}'
    "}"
)
mixed_lmm_message_instance: LMMCompletionMessage = LMMCompletionMessage(
    role="assistant",
    content=(
        AudioURLContent(audio_url="https://miquido.com/audio"),
        "string",
        ImageURLContent(image_url="https://miquido.com/image"),
        "content",
    ),
)
mixed_lmm_message_json: str = (
    '{"role": "assistant",'
    ' "content": ['
    '{"audio_url": "https://miquido.com/audio", "audio_transcription": null},'
    ' "string",'
    ' {"image_url": "https://miquido.com/image", "image_description": null},'
    ' "content"'
    "]}"
)


def test_llm_message_decoding() -> None:
    assert LMMCompletionMessage.from_json(basic_lmm_message_json) == basic_lmm_message_instance
    assert LMMCompletionMessage.from_json(image_lmm_message_json) == image_lmm_message_instance
    assert LMMCompletionMessage.from_json(audio_lmm_message_json) == audio_lmm_message_instance
    assert LMMCompletionMessage.from_json(mixed_lmm_message_json) == mixed_lmm_message_instance


def test_llm_message_encoding() -> None:
    assert basic_lmm_message_instance.as_json() == basic_lmm_message_json
    assert image_lmm_message_instance.as_json() == image_lmm_message_json
    assert audio_lmm_message_instance.as_json() == audio_lmm_message_json
    assert mixed_lmm_message_instance.as_json() == mixed_lmm_message_json
