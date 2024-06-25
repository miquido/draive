import json
from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, Required, TypedDict
from uuid import UUID

from draive import (
    MISSING,
    AudioURLContent,
    ConversationMessage,
    DataModel,
    Field,
    ImageURLContent,
    Missing,
    MultimodalContent,
)


def invalid(value: str) -> None:
    if value != "valid":
        raise ValueError()


class ExampleNestedModel(DataModel):
    nested_alias: str = Field(aliased="string", default="default")
    more: list[int] | None = None


class ExampleModel(DataModel):
    string: str = "default"
    number: int = Field(aliased="alias", description="description", default=1)
    none_default: int | None = Field(default=None)
    value_default: int = Field(default=9)
    invalid: str = Field(verifier=invalid, default="valid")
    nested: ExampleNestedModel = Field(aliased="answer", default=ExampleNestedModel())
    full: Literal["A", "B"] | list[int] | str | bool | None = Field(
        aliased="all",
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
    assert json.loads(ExampleModel(**values_dict).as_json()) == values_dict


def test_validated_passes_with_default_values() -> None:
    ExampleModel()


class DatetimeModel(DataModel):
    value: datetime


datetime_model_instance: DatetimeModel = DatetimeModel(
    value=datetime.fromtimestamp(0, UTC),
)
datetime_model_json: str = '{"value": "1970-01-01T00:00:00+00:00"}'


def test_datetime_encoding() -> None:
    assert datetime_model_instance.as_json() == datetime_model_json


def test_datetime_decoding() -> None:
    assert DatetimeModel.from_json(datetime_model_json) == datetime_model_instance


class UUIDModel(DataModel):
    value: UUID


uuid_model_instance: UUIDModel = UUIDModel(
    value=UUID(hex="0cf728c0369348e78552e8d86d35e8b0"),
)
uuid_model_json: str = '{"value": "0cf728c0369348e78552e8d86d35e8b0"}'


def test_uuid_encoding() -> None:
    assert uuid_model_instance.as_json() == uuid_model_json


def test_uuid_decoding() -> None:
    assert UUIDModel.from_json(uuid_model_json) == uuid_model_instance


class MissingModel(DataModel):
    value: Missing = MISSING


missing_model_instance: MissingModel = MissingModel(
    value=MISSING,
)
missing_model_json: str = '{"value": null}'


def test_missing_encoding() -> None:
    assert missing_model_instance.as_json() == missing_model_json


def test_missing_decoding() -> None:
    assert MissingModel.from_json(missing_model_json) == missing_model_instance


class ExampleTypedDictNested(DataModel):
    value: int


class ExampleTypedDict(TypedDict, total=False):
    dict_int: Required[int]
    dict_str: NotRequired[str | None]
    dict_nested: Required[ExampleTypedDictNested]


class ExampleTotalTypedDict(TypedDict, total=True):
    total: bool


class TypedDictModel(DataModel):
    value: ExampleTypedDict
    total_value: ExampleTotalTypedDict


typed_dict_model_instance: TypedDictModel = TypedDictModel(
    value={
        "dict_int": 42,
        "dict_str": "answer",
        "dict_nested": ExampleTypedDictNested(
            value=21,
        ),
    },
    total_value={"total": True},
)
typed_dict_model_json: str = (
    '{"value":'
    ' {"dict_int": 42, "dict_str": "answer", "dict_nested":'
    ' {"value": 21}'
    "},"
    ' "total_value": {"total": true}'
    "}"
)
typed_dict_not_required_model_instance: TypedDictModel = TypedDictModel(
    value={
        "dict_int": 42,
        "dict_nested": ExampleTypedDictNested(
            value=21,
        ),
    },
    total_value={"total": True},
)
typed_dict_not_required_model_json: str = (
    '{"value":'
    ' {"dict_int": 42, "dict_nested":'
    ' {"value": 21}},'
    ' "total_value": {"total": true}}'
)


def test_typed_dict_encoding() -> None:
    assert typed_dict_model_instance.as_json() == typed_dict_model_json
    assert typed_dict_not_required_model_instance.as_json() == typed_dict_not_required_model_json


def test_typed_dict_decoding() -> None:
    assert TypedDictModel.from_json(typed_dict_model_json) == typed_dict_model_instance
    assert (
        TypedDictModel.from_json(typed_dict_not_required_model_json)
        == typed_dict_not_required_model_instance
    )


class BasicsModel(DataModel):
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
    ' "floating_dict": {"a": 65.0, "b": 66.0, "c": 67.5},'
    ' "optional": "some",'
    ' "none": null}'
)


def test_basic_encoding() -> None:
    assert basic_model_instance.as_json() == basic_model_json


def test_basic_decoding() -> None:
    assert BasicsModel.from_json(basic_model_json) == basic_model_instance


basic_conversation_message_instance: ConversationMessage = ConversationMessage(
    identifier="identifier",
    role="model",
    content=MultimodalContent.of("string"),
)
basic_conversation_message_json: str = (
    "{"
    '"identifier": "identifier", '
    '"role": "model", '
    '"author": null, '
    '"created": null, '
    '"content": {'
    '"parts": [{"text": "string"}]'
    "}}"
)

image_conversation_message_instance: ConversationMessage = ConversationMessage(
    identifier="identifier",
    role="model",
    content=MultimodalContent.of(ImageURLContent(image_url="https://miquido.com/image")),
)
image_conversation_message_json: str = (
    "{"
    '"identifier": "identifier", '
    '"role": "model", '
    '"author": null, '
    '"created": null, '
    '"content": {'
    '"parts": ['
    '{"mime_type": null, "image_url": "https://miquido.com/image", "image_description": null}'
    "]"
    "}}"
)
audio_conversation_message_instance: ConversationMessage = ConversationMessage(
    identifier="identifier",
    role="model",
    content=MultimodalContent.of(AudioURLContent(audio_url="https://miquido.com/audio")),
)
audio_conversation_message_json: str = (
    "{"
    '"identifier": "identifier", '
    '"role": "model", '
    '"author": null, '
    '"created": null, '
    '"content": {'
    '"parts": ['
    '{"mime_type": null, "audio_url": "https://miquido.com/audio", "audio_transcription": null}'
    "]"
    "}}"
)
mixed_conversation_message_instance: ConversationMessage = ConversationMessage(
    identifier="identifier",
    role="model",
    content=MultimodalContent.of(
        AudioURLContent(audio_url="https://miquido.com/audio"),
        "string",
        ImageURLContent(image_url="https://miquido.com/image"),
        "content",
    ),
)
mixed_conversation_message_json: str = (
    "{"
    '"identifier": "identifier", '
    '"role": "model", '
    '"author": null, '
    '"created": null, '
    '"content": {"parts": ['
    '{"mime_type": null, "audio_url": "https://miquido.com/audio", "audio_transcription": null}, '
    '{"text": "string"}, '
    '{"mime_type": null, "image_url": "https://miquido.com/image", "image_description": null}, '
    '{"text": "content"}'
    "]}}"
)


def test_llm_message_decoding() -> None:
    assert (
        ConversationMessage.from_json(basic_conversation_message_json)
        == basic_conversation_message_instance
    )
    assert (
        ConversationMessage.from_json(image_conversation_message_json)
        == image_conversation_message_instance
    )
    assert (
        ConversationMessage.from_json(audio_conversation_message_json)
        == audio_conversation_message_instance
    )
    assert (
        ConversationMessage.from_json(mixed_conversation_message_json)
        == mixed_conversation_message_instance
    )


def test_llm_message_encoding() -> None:
    assert basic_conversation_message_instance.as_json() == basic_conversation_message_json
    assert image_conversation_message_instance.as_json() == image_conversation_message_json
    assert audio_conversation_message_instance.as_json() == audio_conversation_message_json
    assert mixed_conversation_message_instance.as_json() == mixed_conversation_message_json


class AnyModelNested(DataModel):
    answer: int


class AnyModel(DataModel):
    any_model: DataModel


any_model_instance: AnyModel = AnyModel(
    any_model=AnyModelNested(
        answer=42,
    ),
)
any_model_json: str = '{"any_model": {"answer": 42}}'


def test_any_encoding() -> None:
    assert any_model_instance.as_json() == any_model_json


def test_any_dict_decoding() -> None:
    assert DataModel.from_json(any_model_json).as_dict() == any_model_instance.as_dict()
