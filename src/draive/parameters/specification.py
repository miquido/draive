from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from enum import IntEnum, StrEnum
from types import EllipsisType, NoneType, UnionType
from typing import (
    Any,
    Final,
    Literal,
    NotRequired,
    Required,
    TypedDict,
    Union,
    cast,
    final,
    is_typeddict,
)
from uuid import UUID

from haiway import Missing
from haiway.state import AttributeAnnotation

from draive.commons import Meta
from draive.parameters.types import ParameterValidation, ParameterValidationContext
from draive.parameters.validation import ParameterValidator

__all__ = (
    "ParameterSpecification",
    "ParametersSpecification",
    "parameter_specification",
    "validated_specification",
)


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
    description: NotRequired[str]
    required: NotRequired[Sequence[str]]


@final
class ParameterObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[Mapping[str, "ParameterSpecification"]]
    title: NotRequired[str]
    description: NotRequired[str]
    required: NotRequired[Sequence[str]]


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
    ParameterUnionSpecification
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


def validated_specification(
    specification: dict[str, Any],
    /,
) -> ParameterObjectSpecification:
    with ParameterValidationContext().scope("specification") as validation_context:
        return _specification_validation(
            specification,
            context=validation_context,
        )


_specification_validation: Final[ParameterValidation[ParameterObjectSpecification]] = (
    ParameterValidator.of_typed_dict(ParameterObjectSpecification)
)


def parameter_specification(  # noqa: PLR0911
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if specification := SPECIFICATIONS.get(annotation.origin):
        return specification(annotation, description)

    elif hasattr(annotation.origin, "__PARAMETERS_SPECIFICATION__"):
        if description := description:
            return cast(
                ParameterSpecification,
                {
                    **annotation.origin.__PARAMETERS_SPECIFICATION__,
                    "description": description,
                },
            )

        else:
            return annotation.origin.__PARAMETERS_SPECIFICATION__

    elif is_typeddict(annotation.origin):
        return _prepare_specification_of_typed_dict(
            annotation,
            description=description,
        )

    elif issubclass(annotation.origin, IntEnum):
        if description := description:
            return {
                "type": "integer",
                "enum": list(annotation.origin),
                "description": description,
            }

        else:
            return {
                "type": "integer",
                "enum": list(annotation.origin),
            }

    elif issubclass(annotation.origin, StrEnum):
        if description := description:
            return {
                "type": "string",
                "enum": list(annotation.origin),
                "description": description,
            }

        else:
            return {
                "type": "string",
                "enum": list(annotation.origin),
            }

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
    if all(isinstance(element, str) for element in annotation.arguments):
        if description := description:
            return {
                "type": "string",
                "enum": annotation.arguments,
                "description": description,
            }

        else:
            return {
                "type": "string",
                "enum": annotation.arguments,
            }
    else:
        raise TypeError(f"Unsupported type annotation: {annotation}")


def _prepare_specification_of_sequence(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "array",
            "items": parameter_specification(
                annotation.arguments[0],
                description=None,
            ),
            "description": description,
        }

    else:
        return {
            "type": "array",
            "items": parameter_specification(
                annotation.arguments[0],
                description=None,
            ),
        }


def _prepare_specification_of_mapping(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "type": "object",
            "additionalProperties": parameter_specification(
                annotation.arguments[1],
                description=None,
            ),
            "description": description,
        }

    else:
        return {
            "type": "object",
            "additionalProperties": parameter_specification(
                annotation.arguments[1],
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
    if (
        annotation.arguments[-1].origin == Ellipsis
        or annotation.arguments[-1].origin == EllipsisType
    ):
        if description := description:
            return {
                "type": "array",
                "items": parameter_specification(annotation.arguments[0], description=None),
                "description": description,
            }

        else:
            return {
                "type": "array",
                "items": parameter_specification(annotation.arguments[0], description=None),
            }

    elif description := description:
        return {
            "type": "array",
            "prefixItems": [
                parameter_specification(element, description=None)
                for element in annotation.arguments
            ],
            "description": description,
        }

    else:
        return {
            "type": "array",
            "prefixItems": [
                parameter_specification(element, description=None)
                for element in annotation.arguments
            ],
        }


def _prepare_specification_of_union(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    if description := description:
        return {
            "oneOf": [
                parameter_specification(cast(AttributeAnnotation, argument), description=None)
                for argument in annotation.arguments
            ],
            "description": description,
        }

    else:
        return {
            "oneOf": [
                parameter_specification(cast(AttributeAnnotation, argument), description=None)
                for argument in annotation.arguments
            ],
        }


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


def _prepare_specification_of_typed_dict(
    annotation: AttributeAnnotation,
    /,
    description: str | None,
) -> ParameterSpecification:
    required: list[str] = []
    properties: dict[str, ParameterSpecification] = {}

    for key, element in annotation.extra["attributes"].items():
        properties[key] = parameter_specification(
            element,
            description=None,
        )

        if getattr(element, "required", True):
            required.append(key)

    if description := description:
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "description": description,
        }

    else:
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


SPECIFICATIONS: Mapping[
    Any, Callable[[AttributeAnnotation, str | None], ParameterSpecification]
] = {
    Any: _prepare_specification_of_any,
    NoneType: _prepare_specification_of_none,
    Missing: _prepare_specification_of_missing,
    bool: _prepare_specification_of_bool,
    int: _prepare_specification_of_int,
    float: _prepare_specification_of_float,
    str: _prepare_specification_of_str,
    tuple: _prepare_specification_of_tuple,
    Literal: _prepare_specification_of_literal,
    Sequence: _prepare_specification_of_sequence,
    Mapping: _prepare_specification_of_mapping,
    Meta: _prepare_specification_of_meta,
    UUID: _prepare_specification_of_uuid,
    date: _prepare_specification_of_date,
    datetime: _prepare_specification_of_datetime,
    time: _prepare_specification_of_time,
    Union: _prepare_specification_of_union,
    UnionType: _prepare_specification_of_union,
}
