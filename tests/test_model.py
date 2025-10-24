import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from copy import copy, deepcopy
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, Required, TypedDict
from uuid import UUID, uuid4

from haiway import Alias, Default, Description, Validator
from pytest import mark, raises

from draive import (
    MISSING,
    DataModel,
    GenerativeModel,
    Missing,
    ModelOutput,
    ModelToolsDeclaration,
    MultimodalContent,
    TextContent,
    ValidationError,
    ctx,
)
from draive.conversation import ConversationMessage
from draive.resources import ResourceContent, ResourceReference


def invalid_value(value: str) -> str:
    if value != "valid":
        raise ValueError()

    return value


class ExampleNestedModel(DataModel):
    nested_alias: Annotated[str, Alias("string")] = "default"
    more: Sequence[int] | None = None


class ExampleModel(DataModel):
    string: str = "default"
    number: Annotated[int, Alias("alias"), Description("description")] = 1
    none_default: int | None = None
    value_default: int = 9
    invalid: Annotated[str, Validator(invalid_value)] = "valid"
    nested: Annotated[ExampleNestedModel, Alias("answer")] = Default(
        default_factory=ExampleNestedModel
    )
    full: Annotated[
        Literal["A", "B"] | Sequence[int] | str | bool | None, Alias("all"), Description("complex")
    ] = ""
    default: UUID = Default(default_factory=uuid4)


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
        "default": "12345678123456781234567812345678",
    }
    assert json.loads(ExampleModel(**values_dict).to_json()) == values_dict


def test_validated_passes_with_default_values() -> None:
    ExampleModel()


def test_default_factory_produces_values() -> None:
    assert ExampleModel().default != ExampleModel().default


class DatetimeModel(DataModel):
    value: datetime


datetime_model_instance: DatetimeModel = DatetimeModel(
    value=datetime.fromtimestamp(0, UTC),
)
datetime_model_json: str = '{"value": "1970-01-01T00:00:00+00:00"}'


def test_datetime_encoding() -> None:
    assert datetime_model_instance.to_json() == datetime_model_json


def test_datetime_decoding() -> None:
    assert DatetimeModel.from_json(datetime_model_json) == datetime_model_instance


class UUIDModel(DataModel):
    value: UUID


uuid_model_instance: UUIDModel = UUIDModel(
    value=UUID("0cf728c0369348e78552e8d86d35e8b0"),
)
uuid_model_json: str = '{"value": "0cf728c0369348e78552e8d86d35e8b0"}'


def test_uuid_encoding() -> None:
    assert uuid_model_instance.to_json() == uuid_model_json


def test_uuid_decoding() -> None:
    assert UUIDModel.from_json(uuid_model_json) == uuid_model_instance


class MissingModel(DataModel):
    value: Missing = MISSING


missing_model_instance: MissingModel = MissingModel(
    value=MISSING,
)
missing_model_json: str = "{}"


def test_missing_encoding() -> None:
    assert missing_model_instance.to_json(indent=4) == missing_model_json


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
typed_dict_model_json: str = """\
{
    "value": {
        "dict_int": 42,
        "dict_str": "answer",
        "dict_nested": {
            "value": 21
        }
    },
    "total_value": {
        "total": true
    }
}\
"""
typed_dict_not_required_model_instance: TypedDictModel = TypedDictModel(
    value={
        "dict_int": 42,
        "dict_nested": ExampleTypedDictNested(
            value=21,
        ),
    },
    total_value={"total": True},
)
typed_dict_not_required_model_json: str = """\
{
    "value": {
        "dict_int": 42,
        "dict_nested": {
            "value": 21
        }
    },
    "total_value": {
        "total": true
    }
}\
"""


def test_typed_dict_encoding() -> None:
    assert typed_dict_model_instance.to_json(indent=4) == typed_dict_model_json
    assert (
        typed_dict_not_required_model_instance.to_json(indent=4)
        == typed_dict_not_required_model_json
    )


def test_typed_dict_decoding() -> None:
    assert TypedDictModel.from_json(typed_dict_model_json) == typed_dict_model_instance
    assert (
        TypedDictModel.from_json(typed_dict_not_required_model_json)
        == typed_dict_not_required_model_instance
    )


class BasicsModel(DataModel):
    string: str
    string_list: Sequence[str]
    integer: int
    integer_or_float_list: Sequence[int | float]
    floating: float
    floating_dict: Mapping[str, float]
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
basic_model_json: str = """\
{
    "string": "test",
    "string_list": [
        "basic",
        "list"
    ],
    "integer": 42,
    "integer_or_float_list": [
        12,
        3.14,
        7
    ],
    "floating": 9.99,
    "floating_dict": {
        "a": 65.0,
        "b": 66.0,
        "c": 67.5
    },
    "optional": "some",
    "none": null
}\
"""


def test_basic_encoding() -> None:
    assert basic_model_instance.to_json(indent=4) == basic_model_json


def test_basic_decoding() -> None:
    assert BasicsModel.from_json(basic_model_json) == basic_model_instance


basic_conversation_message_instance: ConversationMessage = ConversationMessage.model(
    identifier=UUID("f6da0a47556744cdb8334d263020907f"),
    created=datetime.fromisoformat("2025-06-03T18:15:58.985599"),
    content=MultimodalContent.of("string"),
)
basic_conversation_message_json: str = """\
{
    "identifier": "f6da0a47556744cdb8334d263020907f",
    "role": "model",
    "created": "2025-06-03T18:15:58.985599",
    "content": {
        "type": "content",
        "parts": [
            {
                "text": "string",
                "meta": {}
            }
        ]
    },
    "meta": {}
}\
"""

media_url_conversation_message_instance: ConversationMessage = ConversationMessage.model(
    identifier=UUID("f6da0a47556744cdb8334d263020907f"),
    created=datetime.fromisoformat("2025-06-03T18:15:58.985599"),
    content=MultimodalContent.of(
        ResourceReference.of(uri="https://miquido.com/image", mime_type="image/png")
    ),
)
media_url_conversation_message_json: str = """\
{
    "identifier": "f6da0a47556744cdb8334d263020907f",
    "role": "model",
    "created": "2025-06-03T18:15:58.985599",
    "content": {
        "type": "content",
        "parts": [
            {
                "uri": "https://miquido.com/image",
                "mime_type": "image/png",
                "meta": {
                    "name": null,
                    "description": null
                }
            }
        ]
    },
    "meta": {}
}\
"""

media_data_conversation_message_instance: ConversationMessage = ConversationMessage.model(
    identifier=UUID("f6da0a47556744cdb8334d263020907f"),
    created=datetime.fromisoformat("2025-06-03T18:15:58.985599"),
    content=MultimodalContent.of(ResourceContent.of(b"image_data", mime_type="image/png")),
)
media_data_conversation_message_json: str = """\
{
    "identifier": "f6da0a47556744cdb8334d263020907f",
    "role": "model",
    "created": "2025-06-03T18:15:58.985599",
    "content": {
        "type": "content",
        "parts": [
            {
                "data": "aW1hZ2VfZGF0YQ==",
                "mime_type": "image/png",
                "meta": {}
            }
        ]
    },
    "meta": {}
}\
"""


def test_message_decoding() -> None:
    assert (
        ConversationMessage.from_json(basic_conversation_message_json)
        == basic_conversation_message_instance
    )
    assert (
        ConversationMessage.from_json(media_url_conversation_message_json)
        == media_url_conversation_message_instance
    )
    assert (
        ConversationMessage.from_json(media_data_conversation_message_json)
        == media_data_conversation_message_instance
    )


def test_message_encoding() -> None:
    assert basic_conversation_message_instance.to_json(indent=4) == basic_conversation_message_json
    assert (
        media_url_conversation_message_instance.to_json(indent=4)
        == media_url_conversation_message_json
    )
    assert (
        media_data_conversation_message_instance.to_json(indent=4)
        == media_data_conversation_message_json
    )


class AnyModelNested(DataModel):
    answer: int


class AnyModel(DataModel):
    any_model: DataModel


any_model_instance: AnyModel = AnyModel(
    any_model=AnyModelNested(
        answer=42,
    ),
)
any_model_json: str = """\
{
    "any_model": {
        "answer": 42
    }
}\
"""


def test_any_encoding() -> None:
    assert any_model_instance.to_json(indent=4) == any_model_json


def test_any_dict_decoding() -> None:
    assert DataModel.from_json(any_model_json).to_mapping() == any_model_instance.to_mapping()


def test_generic_subtypes_validation() -> None:
    class NestedGeneric[T](DataModel):
        value: T

    class Generic[T](DataModel):
        nested: NestedGeneric[T]

    class Container(DataModel):
        generic: Generic[str]

    assert isinstance(Generic[str](nested=NestedGeneric[str](value="ok")), Generic)
    assert isinstance(Generic(nested=NestedGeneric[str](value="ok")), Generic)
    assert isinstance(Generic(nested=NestedGeneric(value="ok")), Generic)

    assert isinstance(Generic[str](nested=NestedGeneric[str](value="ok")), Generic[Any])
    assert isinstance(Generic(nested=NestedGeneric[str](value="ok")), Generic[Any])
    assert isinstance(Generic[str](nested=NestedGeneric(value="ok")), Generic[Any])
    assert isinstance(Generic(nested=NestedGeneric(value="ok")), Generic[Any])

    assert isinstance(Generic[str](nested=NestedGeneric[str](value="ok")), Generic[str])
    assert isinstance(Generic(nested=NestedGeneric[str](value="ok")), Generic[str])
    assert isinstance(Generic[str](nested=NestedGeneric(value="ok")), Generic[str])
    assert isinstance(Generic(nested=NestedGeneric(value="ok")), Generic[str])

    with raises(ValidationError):
        _ = Generic[int](nested=NestedGeneric[str](value="ok"))

    with raises(ValidationError):
        _ = Container(generic=Generic(nested=NestedGeneric(value=42)))

    with raises(ValidationError):
        _ = Container(generic=Generic[int](nested=NestedGeneric[str](value="ok")))

    # not raises
    _ = Container(generic=Generic(nested=NestedGeneric(value="ok")))


def test_copying_leaves_same_object() -> None:
    class Nested(DataModel):
        string: str

    class Copied(DataModel):
        string: str
        nested: Nested

    origin = Copied(string="42", nested=Nested(string="answer"))
    assert copy(origin) is origin
    assert deepcopy(origin) is origin


@mark.asyncio
async def test_generative_model_completion_multi_modal_selection() -> None:
    text_part: TextContent = TextContent.of("hello")
    image_part: ResourceReference = ResourceReference.of(
        "image://example",
        mime_type="image/png",
    )
    audio_part: ResourceReference = ResourceReference.of(
        "audio://example",
        mime_type="audio/mpeg",
    )

    async def _non_stream_generating(**_: Any) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of(text_part, image_part, audio_part))

    def generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[Any],
        output: Any,
        stream: bool = False,
        **_: Any,
    ) -> AsyncGenerator[Any] | Any:
        assert not stream
        return _non_stream_generating()

    async with ctx.scope(
        "test.multi.selection",
        GenerativeModel(generating=generating),
    ):
        result: ModelOutput = await GenerativeModel.completion(
            instructions="",
            tools=ModelToolsDeclaration.none,
            context=(),
            output={"text", "image"},
        )

    parts = result.content.parts
    assert len(parts) == 2
    assert isinstance(parts[0], TextContent)
    assert isinstance(parts[1], ResourceReference)
    assert (parts[1].mime_type or "").startswith("image")


@mark.asyncio
async def test_generative_model_stream_multi_modal_selection() -> None:
    text_part: TextContent = TextContent.of("stream")
    image_part: ResourceReference = ResourceReference.of(
        "image://example",
        mime_type="image/png",
    )
    audio_part: ResourceReference = ResourceReference.of(
        "audio://example",
        mime_type="audio/mpeg",
    )

    async def _non_stream_generating(**_: Any) -> ModelOutput:
        return ModelOutput.of(MultimodalContent.of(text_part, image_part, audio_part))

    def generating(
        *,
        instructions: str,
        tools: ModelToolsDeclaration,
        context: Sequence[Any],
        output: Any,
        stream: bool = False,
        **_: Any,
    ) -> AsyncGenerator[Any] | Any:
        if stream:

            async def iterator() -> AsyncGenerator[Any]:
                yield text_part
                yield image_part
                yield audio_part

            return iterator()

        return _non_stream_generating()

    async with ctx.scope(
        "test.multi.selection.stream",
        GenerativeModel(generating=generating),
    ):
        stream = await GenerativeModel.completion(
            instructions="",
            tools=ModelToolsDeclaration.none,
            context=(),
            output={"text", "image"},
            stream=True,
        )

        parts: list[Any] = [chunk async for chunk in stream]

    assert len(parts) == 2
    assert isinstance(parts[0], TextContent)
    assert isinstance(parts[1], ResourceReference)
    assert (parts[1].mime_type or "").startswith("image")
