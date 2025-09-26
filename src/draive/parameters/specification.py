from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
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
    type: Required[Sequence[Literal["string", "number", "integer", "boolean", "null"]]]
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
    if isinstance(annotation, AliasAttribute):
        return parameter_specification(annotation.resolved, description)

    if specification := SPECIFICATIONS.get(type(annotation)):
        return specification(annotation, description)

    elif hasattr(annotation.base, "__PARAMETERS_SPECIFICATION__"):
        if description := description:
            return cast(
                ParameterSpecification,
                {
                    **annotation.base.__PARAMETERS_SPECIFICATION__,
                    "description": description,
                },
            )

        else:
            return annotation.base.__PARAMETERS_SPECIFICATION__

    else:
        raise TypeError(f"Unsupported type annotation: {annotation}")


def _prepare_specification_of_any(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "object",
            "additionalProperties": True,
            "description": description,
        }

    else:
        return {
            "type": "object",
            "additionalProperties": True,
        }


def _prepare_specification_of_none(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "null",
            "description": description,
        }

    else:
        return {
            "type": "null",
        }


def _prepare_specification_of_missing(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "null",
            "description": description,
        }

    else:
        return {
            "type": "null",
        }


def _prepare_specification_of_literal(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    literal = cast(LiteralAttribute, annotation)

    if all(isinstance(element, str) for element in literal.values):
        if description := description:
            return {
                "type": "string",
                "enum": literal.values,
                "description": description,
            }

        else:
            return {
                "type": "string",
                "enum": literal.values,
            }

    else:
        raise TypeError(f"Unsupported type annotation: {annotation}")


def _prepare_specification_of_sequence(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    sequence = cast(SequenceAttribute, annotation)

    if description := description:
        return {
            "type": "array",
            "items": parameter_specification(
                sequence.values,
                description=None,
            ),
            "description": description,
        }

    else:
        return {
            "type": "array",
            "items": parameter_specification(
                sequence.values,
                description=None,
            ),
        }


def _prepare_specification_of_mapping(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    mapping = cast(MappingAttribute, annotation)

    if description := description:
        return {
            "type": "object",
            "additionalProperties": parameter_specification(
                mapping.values,
                description=None,
            ),
            "description": description,
        }

    else:
        return {
            "type": "object",
            "additionalProperties": parameter_specification(
                mapping.values,
                description=None,
            ),
        }


def _prepare_specification_of_meta(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    # TODO: prepare actual property types declaration
    if description := description:
        return {
            "type": "object",
            "additionalProperties": True,
            "description": description,
        }

    else:
        return {
            "type": "object",
            "additionalProperties": True,
        }


def _prepare_specification_of_tuple(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    tuple_attribute = cast(TupleAttribute, annotation)

    if description := description:
        return {
            "type": "array",
            "prefixItems": [
                parameter_specification(element, description=None)
                for element in tuple_attribute.values
            ],
            "description": description,
        }

    else:
        return {
            "type": "array",
            "prefixItems": [
                parameter_specification(element, description=None)
                for element in tuple_attribute.values
            ],
        }


def _prepare_specification_of_union(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    union = cast(UnionAttribute, annotation)

    compressed_alternatives: list[Literal["string", "number", "integer", "boolean", "null"]] = []
    alternatives: list[ParameterSpecification] = []
    for argument in union.alternatives:
        specification: ParameterSpecification = parameter_specification(
            cast(AttributeAnnotation, argument),
            description=None,
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
                pass  # skip - type is more complex and can't be compressed

    if description := description:
        if len(compressed_alternatives) == len(alternatives):
            # prefer comperessed when equivalent representation is available
            return ParameterAlternativesSpecification(
                type=compressed_alternatives,
                description=description,
            )

        return {
            "oneOf": alternatives,
            "description": description,
        }

    else:
        if len(compressed_alternatives) == len(alternatives):
            # prefer comperessed when equivalent representation is available
            return ParameterAlternativesSpecification(
                type=compressed_alternatives,
            )

        return {"oneOf": alternatives}


def _prepare_specification_of_bool(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "boolean",
            "description": description,
        }

    else:
        return {
            "type": "boolean",
        }


def _prepare_specification_of_int(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "integer",
            "description": description,
        }

    else:
        return {
            "type": "integer",
        }


def _prepare_specification_of_float(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "number",
            "description": description,
        }

    else:
        return {
            "type": "number",
        }


def _prepare_specification_of_str(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "string",
            "description": description,
        }

    else:
        return {
            "type": "string",
        }


def _prepare_specification_of_str_enum(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    enum_attribute = cast(StrEnumAttribute, annotation)
    enum_values: list[str] = [member.value for member in enum_attribute.base]

    specification: ParameterStringEnumSpecification = {
        "type": "string",
        "enum": enum_values,
    }

    if description := description:
        specification["description"] = description

    return specification


def _prepare_specification_of_int_enum(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    enum_attribute = cast(IntEnumAttribute, annotation)
    enum_values: list[int] = [int(member.value) for member in enum_attribute.base]

    specification: ParameterIntegerEnumSpecification = {
        "type": "integer",
        "enum": enum_values,
    }

    if description := description:
        specification["description"] = description

    return specification


def _prepare_specification_of_uuid(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "string",
            "format": "uuid",
            "description": description,
        }

    else:
        return {
            "type": "string",
            "format": "uuid",
        }


def _prepare_specification_of_date(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "string",
            "format": "date",
            "description": description,
        }

    else:
        return {
            "type": "string",
            "format": "date",
        }


def _prepare_specification_of_datetime(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "string",
            "format": "date-time",
            "description": description,
        }

    else:
        return {
            "type": "string",
            "format": "date-time",
        }


def _prepare_specification_of_time(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "string",
            "format": "time",
            "description": description,
        }

    else:
        return {
            "type": "string",
            "format": "time",
        }


def _prepare_specification_of_custom(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    base: Any = annotation.base
    if hasattr(base, "__PARAMETERS_SPECIFICATION__"):
        specification = cast(ParameterSpecification, base.__PARAMETERS_SPECIFICATION__)
        if description := description:
            return cast(
                ParameterSpecification,
                {
                    **cast(dict[str, Any], specification),
                    "description": description,
                },
            )

        return specification

    if base is Meta:
        if description := description:
            return {
                "type": "object",
                "additionalProperties": True,
                "description": description,
            }

        else:
            return {
                "type": "object",
                "additionalProperties": True,
            }

    raise TypeError(f"Unsupported custom attribute: {annotation}")


def _prepare_specification_of_validable(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    validable = cast(ValidableAttribute, annotation)
    return parameter_specification(validable.attribute, description)


def _prepare_specification_of_typed_dict(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    typed_dict = cast(TypedDictAttribute, annotation)

    required: list[str] = []
    properties: dict[str, ParameterSpecification] = {}

    for key, element in typed_dict.attributes.items():
        properties[key] = parameter_specification(
            element,
            description=None,
        )

        if NotRequired in element.annotations:
            continue  # not required

        # default to required when annotations are missing
        required.append(key)

    if description := description:
        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": required,
            "description": description,
        }

    else:
        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": required,
        }


SPECIFICATIONS: Mapping[
    type[AttributeAnnotation],
    Callable[[AttributeAnnotation, str | None], ParameterSpecification],
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
