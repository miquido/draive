from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import (
    Literal,
    NotRequired,
    Required,
    TypedDict,
    cast,
    final,
)

from haiway import AttributeAnnotation, Meta
from haiway.attributes.annotations import (
    AliasAttribute,
    AnyAttribute,
    BoolAttribute,
    CustomAttribute,
    DatetimeAttribute,
    FloatAttribute,
    IntegerAttribute,
    IntEnumAttribute,
    LiteralAttribute,
    MappingAttribute,
    MissingAttribute,
    NoneAttribute,
    SequenceAttribute,
    StrEnumAttribute,
    StringAttribute,
    TimeAttribute,
    TupleAttribute,
    TypedDictAttribute,
    UnionAttribute,
    UUIDAttribute,
    ValidableAttribute,
)

__all__ = (
    "ParameterSpecification",
    "ParametersSpecification",
    "ToolParametersSpecification",
    "parameter_specification",
)


@final
class ParameterAlternativesSpecification(TypedDict, total=False):
    type: Required[
        Sequence[Literal["string", "number", "integer", "boolean", "null", "object", "array"]]
    ]
    description: NotRequired[str]


@final
class ParameterNoneSpecification(TypedDict, total=False):
    type: Required[Literal["null"]]
    description: NotRequired[str]


@final
class ParameterBoolSpecification(TypedDict, total=False):
    type: Required[Literal["boolean"]]
    description: NotRequired[str]


@final
class ParameterIntegerSpecification(TypedDict, total=False):
    type: Required[Literal["integer"]]
    description: NotRequired[str]


@final
class ParameterNumberSpecification(TypedDict, total=False):
    type: Required[Literal["number"]]
    description: NotRequired[str]


@final
class ParameterStringSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]
    description: NotRequired[str]
    format: NotRequired[
        Literal[
            "uri",
            "uuid",
            "date",
            "time",
            "date-time",
        ]
    ]


@final
class ParameterStringEnumSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]
    enum: Required[Sequence[str]]
    description: NotRequired[str]


@final
class ParameterIntegerEnumSpecification(TypedDict, total=False):
    type: Required[Literal["integer"]]
    enum: Required[Sequence[int]]
    description: NotRequired[str]


@final
class ParameterNumberEnumSpecification(TypedDict, total=False):
    type: Required[Literal["number"]]
    enum: Required[Sequence[float]]
    description: NotRequired[str]


@final
class ParameterUnionSpecification(TypedDict, total=False):
    oneOf: Required[Sequence["ParameterSpecification"]]
    description: NotRequired[str]


@final
class ParameterArraySpecification(TypedDict, total=False):
    type: Required[Literal["array"]]
    items: NotRequired["ParameterSpecification"]
    description: NotRequired[str]


@final
class ParameterTupleSpecification(TypedDict, total=False):
    type: Required[Literal["array"]]
    prefixItems: Required[Sequence["ParameterSpecification"]]
    description: NotRequired[str]


@final
class ParameterDictSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    additionalProperties: Required["ParameterSpecification"]
    required: NotRequired[Sequence[str]]
    description: NotRequired[str]


@final
class ParameterObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[Mapping[str, "ParameterSpecification"]]
    additionalProperties: Required[Literal[False]]
    required: NotRequired[Sequence[str]]
    title: NotRequired[str]
    description: NotRequired[str]


@final
class ParameterAnyObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    additionalProperties: Required[Literal[True]]
    description: NotRequired[str]


ReferenceParameterSpecification = TypedDict(
    "ReferenceParameterSpecification",
    {
        "$ref": Required[str],
        "description": NotRequired[str],
    },
    total=False,
)


type ParameterSpecification = (
    ParameterAlternativesSpecification
    | ParameterUnionSpecification
    | ParameterNoneSpecification
    | ParameterStringEnumSpecification
    | ParameterStringSpecification
    | ParameterIntegerEnumSpecification
    | ParameterIntegerSpecification
    | ParameterNumberEnumSpecification
    | ParameterNumberSpecification
    | ParameterBoolSpecification
    | ParameterTupleSpecification
    | ParameterArraySpecification
    | ParameterObjectSpecification
    | ParameterDictSpecification
    | ParameterAnyObjectSpecification
    | ReferenceParameterSpecification
)

type ParametersSpecification = ParameterObjectSpecification


@final
class ToolParametersSpecification(TypedDict, total=False):
    """Strict object schema used for tool parameters.

    Requires ``additionalProperties: False`` so providers cannot accept unspecified args.
    """

    type: Required[Literal["object"]]
    properties: Required[Mapping[str, ParameterSpecification]]
    title: NotRequired[str]
    description: NotRequired[str]
    required: NotRequired[Sequence[str]]
    additionalProperties: Required[Literal[False]]


def parameter_specification(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    return _with_description(
        _parameter_specification(
            annotation,
            recursion_guard={},
        ),
        description=description,
    )


class _RecursionGuard:
    __slots__ = (
        "annotation",
        "requested",
    )

    def __init__(
        self,
        annotation: AttributeAnnotation,
    ) -> None:
        self.annotation: AttributeAnnotation = annotation
        self.requested: bool = False

    @property
    def name(self) -> str:
        return self.annotation.name


def _parameter_specification(  # noqa: PLR0911
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    key: int = id(annotation)

    if guard := recursion_guard.get(key):
        guard.requested = True
        return {"$ref": f"#{guard.name}"}

    elif isinstance(annotation, AliasAttribute):
        recursion_guard[key] = _RecursionGuard(annotation)
        specification: ParameterSpecification = _parameter_specification(
            annotation.resolved,
            recursion_guard,
        )

        if recursion_guard[key].requested:
            return _with_identifier(
                specification,
                identifier=annotation.name,
            )

        return specification

    elif hasattr(annotation.base, "__PARAMETERS_SPECIFICATION__"):
        recursion_guard[key] = _RecursionGuard(annotation)
        specification: ParameterSpecification = cast(
            ParameterSpecification,
            annotation.base.__PARAMETERS_SPECIFICATION__,
        )
        if recursion_guard[key].requested:
            return _with_identifier(
                specification,
                identifier=annotation.name,
            )

        return specification

    elif specification_factory := SPECIFICATIONS.get(type(annotation)):
        guard = recursion_guard.setdefault(
            key,
            _RecursionGuard(annotation),
        )

        specification: ParameterSpecification
        try:
            specification = specification_factory(
                annotation,
                recursion_guard,
            )

        finally:
            if not guard.requested:
                recursion_guard.pop(key, None)

        if guard.requested:
            return _with_identifier(
                specification,
                identifier=guard.name,
            )

        return specification

    else:
        raise TypeError(f"Unsupported type annotation: {annotation}")


def _prepare_specification_of_any(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "object",
        "additionalProperties": True,
    }


def _prepare_specification_of_none(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "null",
    }


def _prepare_specification_of_missing(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "null",
    }


def _prepare_specification_of_literal(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    literal: LiteralAttribute = cast(LiteralAttribute, annotation)

    if all(isinstance(element, str) for element in literal.values):
        return {
            "type": "string",
            "enum": literal.values,
        }

    elif all(isinstance(element, int) for element in literal.values):
        return {
            "type": "integer",
            "enum": literal.values,
        }

    raise TypeError(f"Unsupported literal annotation: {annotation}")


def _prepare_specification_of_sequence(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "array",
        "items": _parameter_specification(
            cast(SequenceAttribute, annotation).values,
            recursion_guard=recursion_guard,
        ),
    }


def _prepare_specification_of_mapping(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "object",
        "additionalProperties": _parameter_specification(
            cast(MappingAttribute, annotation).values,
            recursion_guard=recursion_guard,
        ),
    }


def _prepare_specification_of_meta(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "object",
        "additionalProperties": True,
    }


def _prepare_specification_of_tuple(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    tuple_attribute = cast(TupleAttribute, annotation)
    return {
        "type": "array",
        "prefixItems": [
            _parameter_specification(
                cast(AttributeAnnotation, element),
                recursion_guard=recursion_guard,
            )
            for element in tuple_attribute.values
        ],
    }


def _prepare_specification_of_union(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    compressed_alternatives: list[Literal["string", "number", "integer", "boolean", "null"]] = []
    alternatives: list[ParameterSpecification] = []
    for argument in cast(UnionAttribute, annotation).alternatives:
        specification: ParameterSpecification = _parameter_specification(
            argument,
            recursion_guard=recursion_guard,
        )

        alternatives.append(specification)
        match specification:
            case {"type": "null", **tail} if not tail:
                compressed_alternatives.append("null")

            case {"type": "string", **tail} if not tail:
                compressed_alternatives.append("string")

            case {"type": "number", **tail} if not tail:
                compressed_alternatives.append("number")

            case {"type": "integer", **tail} if not tail:
                compressed_alternatives.append("integer")

            case {"type": "boolean", **tail} if not tail:
                compressed_alternatives.append("boolean")

            case _:
                pass

    if alternatives and len(compressed_alternatives) == len(alternatives):
        return cast(
            ParameterSpecification,
            {
                "type": list(compressed_alternatives),
            },
        )

    return {
        "oneOf": alternatives,
    }


def _prepare_specification_of_bool(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "boolean",
    }


def _prepare_specification_of_int(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "integer",
    }


def _prepare_specification_of_float(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "number",
    }


def _prepare_specification_of_str(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "string",
    }


def _prepare_specification_of_str_enum(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "string",
        "enum": [member.value for member in cast(StrEnumAttribute, annotation).base],
    }


def _prepare_specification_of_int_enum(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "integer",
        "enum": [int(member.value) for member in cast(IntEnumAttribute, annotation).base],
    }


def _prepare_specification_of_uuid(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "string",
        "format": "uuid",
    }


def _prepare_specification_of_date(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "string",
        "format": "date",
    }


def _prepare_specification_of_datetime(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "string",
        "format": "date-time",
    }


def _prepare_specification_of_time(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return {
        "type": "string",
        "format": "time",
    }


def _prepare_specification_of_custom(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    if annotation.base is Meta:
        return {
            "type": "object",
            "additionalProperties": True,
        }

    raise TypeError(f"Unsupported custom attribute: {annotation}")


def _prepare_specification_of_validable(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    return _parameter_specification(
        cast(ValidableAttribute, annotation).attribute,
        recursion_guard=recursion_guard,
    )


def _prepare_specification_of_typed_dict(
    annotation: AttributeAnnotation,
    recursion_guard: MutableMapping[int, _RecursionGuard],
) -> ParameterSpecification:
    typed_dict = cast(TypedDictAttribute, annotation)

    required: list[str] = []
    properties: dict[str, ParameterSpecification] = {}

    for key, element in typed_dict.attributes.items():
        properties[key] = _parameter_specification(
            element,
            recursion_guard=recursion_guard,
        )

        if NotRequired in element.annotations:
            continue

        required.append(key)

    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "required": required,
    }


def _with_description(
    specification: ParameterSpecification,
    description: str | None,
) -> ParameterSpecification:
    if not description:
        return specification

    return cast(
        ParameterSpecification,
        {
            **specification,
            "description": description,
        },
    )


def _with_identifier(
    specification: ParameterSpecification,
    identifier: str,
) -> ParameterSpecification:
    return cast(
        ParameterSpecification,
        {
            **specification,
            "$id": f"#{identifier}",
        },
    )


SPECIFICATIONS: Mapping[
    type[AttributeAnnotation],
    Callable[
        [AttributeAnnotation, MutableMapping[int, _RecursionGuard]],
        ParameterSpecification,
    ],
] = {
    AnyAttribute: _prepare_specification_of_any,
    NoneAttribute: _prepare_specification_of_none,
    MissingAttribute: _prepare_specification_of_missing,
    BoolAttribute: _prepare_specification_of_bool,
    IntegerAttribute: _prepare_specification_of_int,
    FloatAttribute: _prepare_specification_of_float,
    StringAttribute: _prepare_specification_of_str,
    StrEnumAttribute: _prepare_specification_of_str_enum,
    IntEnumAttribute: _prepare_specification_of_int_enum,
    LiteralAttribute: _prepare_specification_of_literal,
    SequenceAttribute: _prepare_specification_of_sequence,
    TupleAttribute: _prepare_specification_of_tuple,
    MappingAttribute: _prepare_specification_of_mapping,
    TypedDictAttribute: _prepare_specification_of_typed_dict,
    UnionAttribute: _prepare_specification_of_union,
    ValidableAttribute: _prepare_specification_of_validable,
    UUIDAttribute: _prepare_specification_of_uuid,
    DatetimeAttribute: _prepare_specification_of_datetime,
    TimeAttribute: _prepare_specification_of_time,
    CustomAttribute: _prepare_specification_of_custom,
}
