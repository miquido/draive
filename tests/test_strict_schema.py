from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from haiway import Meta, State, as_dict

from draive.utils.schema import strict_schema

OPEN_MAP: Mapping[str, Any] = {"type": "object", "additionalProperties": {"type": "string"}}


def _object(
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def test_every_property_becomes_required() -> None:
    converted = strict_schema(
        _object(
            {"a": {"type": "string"}, "b": {"type": "integer"}},
            required=["a"],
        )
    )

    assert converted is not None
    assert converted["required"] == ["a", "b"]
    assert converted["additionalProperties"] is False


def test_open_map_properties_stay_out_of_required() -> None:
    # strict mode strips an open map from the schema and rejects it within `required`
    converted = strict_schema(
        _object(
            {
                "a": {"type": "string"},
                "m": OPEN_MAP,
                "ms": {"type": "array", "items": OPEN_MAP},
            },
            required=["a"],
        )
    )

    assert converted is not None
    assert converted["required"] == ["a"]


def test_nested_objects_are_converted() -> None:
    converted = strict_schema(
        _object(
            {
                "n": _object({"x": {"type": "string"}, "y": {"type": "string"}}, required=["x"]),
                "items": {
                    "type": "array",
                    "items": _object({"p": {"type": "string"}, "q": {"type": "string"}}),
                },
                "either": {
                    "anyOf": [
                        _object({"u": {"type": "string"}, "v": {"type": "string"}}),
                        {"type": "null"},
                    ]
                },
            },
            required=["n"],
        )
    )

    assert converted is not None
    assert converted["required"] == ["n", "items", "either"]
    assert converted["properties"]["n"]["required"] == ["x", "y"]
    assert converted["properties"]["items"]["items"]["required"] == ["p", "q"]
    assert converted["properties"]["either"]["anyOf"][0]["required"] == ["u", "v"]


def test_unsupported_string_format_is_dropped() -> None:
    converted = strict_schema(
        _object(
            {
                "uri": {"type": "string", "format": "uri"},
                "moment": {"type": "string", "format": "date-time"},
            }
        )
    )

    assert converted is not None
    assert "format" not in converted["properties"]["uri"]
    assert converted["properties"]["moment"]["format"] == "date-time"


def test_open_ended_mapping_is_not_convertible() -> None:
    assert (
        strict_schema(_object({"meta": {"type": "object", "additionalProperties": True}})) is None
    )


def test_tuple_is_not_convertible() -> None:
    assert (
        strict_schema(
            _object(
                {
                    "pair": {
                        "type": "array",
                        "prefixItems": [{"type": "string"}, {"type": "integer"}],
                        "items": False,
                    }
                }
            )
        )
        is None
    )


def test_unbounded_array_is_not_convertible() -> None:
    assert strict_schema(_object({"values": {"type": "array"}})) is None


def test_recursive_reference_is_not_convertible() -> None:
    assert (
        strict_schema(
            {
                "$anchor": "Node",
                "type": "object",
                "properties": {"child": {"anyOf": [{"$ref": "#Node"}, {"type": "null"}]}},
                "required": [],
                "additionalProperties": False,
            }
        )
        is None
    )


def test_nesting_beyond_the_limit_is_not_convertible() -> None:
    def nested(depth: int) -> Mapping[str, Any]:
        schema: Mapping[str, Any] = {"type": "string"}
        for _ in range(depth):
            schema = _object({"n": schema})

        return schema

    assert strict_schema(nested(10)) is not None
    assert strict_schema(nested(11)) is None
    # arrays do not add a nesting level of their own
    deep_arrays: Mapping[str, Any] = {"type": "string"}
    for _ in range(20):
        deep_arrays = {"type": "array", "items": deep_arrays}

    assert strict_schema(_object({"n": deep_arrays})) is not None


def test_non_object_root_is_not_convertible() -> None:
    assert strict_schema({"type": "string"}) is None


class _Item(State, serializable=True):
    label: str
    quantity: int = 1
    tags: Sequence[str] = ()


class _Order(State, serializable=True):
    reference: str
    items: Sequence[_Item]
    attributes: Mapping[str, str] = {}
    identifier: UUID | None = None
    moment: datetime | None = None
    total: float | None = None


class _WithMeta(State, serializable=True):
    name: str
    meta: Meta = Meta.empty


class _WithPath(State, serializable=True):
    location: Path


def test_state_specification_converts() -> None:
    converted = strict_schema(as_dict(_Order.__SPECIFICATION__))

    assert converted is not None
    # every field except the open mapping is demanded from the model
    assert converted["required"] == [
        "reference",
        "items",
        "identifier",
        "moment",
        "total",
    ]
    assert converted["properties"]["items"]["items"]["required"] == [
        "label",
        "quantity",
        "tags",
    ]


def test_state_specification_with_open_metadata_falls_back() -> None:
    assert strict_schema(as_dict(_WithMeta.__SPECIFICATION__)) is None


def test_state_specification_with_path_stays_strict() -> None:
    converted = strict_schema(as_dict(_WithPath.__SPECIFICATION__))

    assert converted is not None
    assert "format" not in converted["properties"]["location"]


def test_keywords_outside_the_specification_vocabulary_are_not_convertible() -> None:
    # only the declaration vocabulary is translated, anything else falls back rather
    # than reaching the api - including keywords strict mode would have accepted
    for keyword, value in (
        ("$ref", "#Node"),
        ("$anchor", "Node"),
        ("prefixItems", [{"type": "string"}]),
        ("allOf", [{"type": "string"}]),
        ("not", {"type": "integer"}),
        ("uniqueItems", True),
        ("patternProperties", {"^a": {"type": "string"}}),
        ("minLength", 2),
        ("pattern", "^a+$"),
        ("title", "Name"),
        ("someFutureKeyword", True),
    ):
        assert strict_schema(_object({"p": {"type": "string", keyword: value}})) is None, keyword


def test_declared_descriptions_are_preserved() -> None:
    converted = strict_schema(
        {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["low", "high"], "description": "how soon"},
                "moment": {"type": "string", "format": "date-time"},
            },
            "required": [],
            "additionalProperties": False,
        }
    )

    assert converted is not None
    assert converted["properties"]["priority"]["enum"] == ["low", "high"]
    assert converted["properties"]["priority"]["description"] == "how soon"
    assert converted["properties"]["moment"]["format"] == "date-time"


def test_conversion_never_mutates_its_input() -> None:
    # `as_dict` hands over the live `__SPECIFICATION__` rather than a copy, so a
    # conversion writing in place would corrupt the model class itself
    schema = as_dict(_Order.__SPECIFICATION__)
    before = deepcopy(schema)

    converted = strict_schema(schema)

    assert converted is not None
    assert converted is not schema
    assert schema == before
    assert as_dict(_Order.__SPECIFICATION__) == before
    # the original stays as declared, only the conversion demands every field
    assert list(schema["required"]) == ["reference", "items"]
    assert converted["required"] == ["reference", "items", "identifier", "moment", "total"]


def test_failed_conversion_never_mutates_its_input() -> None:
    schema = as_dict(_WithMeta.__SPECIFICATION__)
    before = deepcopy(schema)

    assert strict_schema(schema) is None
    assert schema == before


def test_scalar_type_alternatives_are_convertible() -> None:
    converted = strict_schema(
        _object(
            {
                "nullable": {"type": ["string", "null"]},
                "either": {"type": ["string", "integer"]},
                "scalars": {"type": ["string", "number", "integer", "boolean", "null"]},
            }
        )
    )

    assert converted is not None
    assert converted["properties"]["either"]["type"] == ["string", "integer"]


def test_object_or_array_type_alternatives_are_not_convertible() -> None:
    # an "object" member of an alternatives array still needs `additionalProperties`
    # and an "array" member needs `items`, neither of which fits there
    for alternatives in (
        ["string", "object"],
        ["string", "array"],
        ["object"],
        ["array"],
        ["string", "number", "integer", "boolean", "object", "array", "null"],
    ):
        assert strict_schema(_object({"p": {"type": alternatives}})) is None, alternatives


class _WithAny(State, serializable=True):
    name: str
    anything: Any = None


class _WithAnyMapping(State, serializable=True):
    name: str
    attributes: Mapping[str, Any] = {}


class _Empty(State, serializable=True):
    pass


def test_any_valued_specifications_fall_back() -> None:
    # both widen to a `type` array spanning objects and arrays
    assert strict_schema(as_dict(_WithAny.__SPECIFICATION__)) is None
    assert strict_schema(as_dict(_WithAnyMapping.__SPECIFICATION__)) is None


def test_fieldless_specification_converts() -> None:
    converted = strict_schema(as_dict(_Empty.__SPECIFICATION__))

    assert converted is not None
    assert converted["properties"] == {}
    assert converted["required"] == []


def test_open_maps_can_be_rejected() -> None:
    # a tool argument object holding an open map has to go unenforced, strict mode
    # strips the map out of it and the model loses whatever it wanted to put there
    for label, properties in (
        ("bare", {"m": OPEN_MAP}),
        ("behind an array", {"ms": {"type": "array", "items": OPEN_MAP}}),
        (
            "nested in an object",
            {"n": _object({"m": OPEN_MAP, "x": {"type": "string"}})},
        ),
        (
            "as a map value",
            {"m": {"type": "object", "additionalProperties": OPEN_MAP}},
        ),
    ):
        schema = _object({"a": {"type": "string"}, **properties})

        assert strict_schema(schema) is not None, label
        assert strict_schema(schema, open_maps=False) is None, label


def test_schemas_without_open_maps_convert_either_way() -> None:
    schema = _object({"a": {"type": "string"}, "b": {"type": "integer"}})

    assert strict_schema(schema) == strict_schema(schema, open_maps=False)
