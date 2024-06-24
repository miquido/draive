import builtins
import datetime
import enum
import types
import typing
import uuid
from collections import abc as collections_abc
from collections.abc import Sequence
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import (
    Any,
    Literal,
    NotRequired,
    Required,
    TypedDict,
    final,
)

from draive.parameters.annotations import resolve_annotation
from draive.utils import MISSING, Missing, not_missing

__all__ = [
    "parameter_specification",
    "ParameterSpecification",
    "ParametersSpecification",
    "ToolFunctionSpecification",
    "ToolSpecification",
]


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
    type: NotRequired[Literal["string"]]
    enum: Required[list[str]]
    description: NotRequired[str]


@final
class ParameterIntegerEnumSpecification(TypedDict, total=False):
    type: NotRequired[Literal["integer"]]
    enum: Required[list[int]]
    description: NotRequired[str]


@final
class ParameterNumberEnumSpecification(TypedDict, total=False):
    type: NotRequired[Literal["number"]]
    enum: Required[list[float]]
    description: NotRequired[str]


ParameterEnumSpecification = (
    ParameterStringEnumSpecification
    | ParameterIntegerEnumSpecification
    | ParameterNumberEnumSpecification
)


@final
class ParameterUnionSpecification(TypedDict, total=False):
    oneOf: Required[list["ParameterSpecification"]]
    description: NotRequired[str]


@final
class ParameterArraySpecification(TypedDict, total=False):
    type: Required[Literal["array"]]
    items: NotRequired["ParameterSpecification"]
    description: NotRequired[str]


@final
class ParameterTupleSpecification(TypedDict, total=False):
    type: Required[Literal["array"]]
    prefixItems: Required[list["ParameterSpecification"]]
    description: NotRequired[str]


@final
class ParameterDictSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    additionalProperties: Required["ParameterSpecification | bool"]
    description: NotRequired[str]
    required: NotRequired[list[str]]


@final
class ParameterObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[dict[str, "ParameterSpecification"]]
    title: NotRequired[str]
    description: NotRequired[str]
    required: NotRequired[list[str]]


ReferenceParameterSpecification = TypedDict(
    "ReferenceParameterSpecification",
    {
        "$ref": Required[str],
        "description": NotRequired[str],
    },
    total=False,
)


ParameterSpecification = (
    ParameterNoneSpecification
    | ParameterBoolSpecification
    | ParameterIntegerSpecification
    | ParameterNumberSpecification
    | ParameterStringSpecification
    | ParameterEnumSpecification
    | ParameterUnionSpecification
    | ParameterArraySpecification
    | ParameterTupleSpecification
    | ParameterDictSpecification
    | ParameterObjectSpecification
    | ReferenceParameterSpecification
)

ParametersSpecification = ParameterObjectSpecification


@final
class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: Required[str]
    parameters: Required[ParametersSpecification]


@final
class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]


def parameter_specification(  # noqa: C901, PLR0912, PLR0915, PLR0911
    annotation: Any,
    description: str | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterSpecification | Missing:
    resolved_origin, resolved_args = resolve_annotation(
        annotation,
        globalns=globalns,
        localns=localns,
    )
    if resolved_origin in recursion_guard:
        # TODO: FIXME: recursive specification is not properly supported
        reference: ReferenceParameterSpecification = {"$ref": resolved_origin.__qualname__}
        if description := description:
            reference["description"] = description
        return reference

    specification: ParameterSpecification
    match resolved_origin:
        case builtins.str:
            specification = {
                "type": "string",
            }

        case builtins.int:
            specification = {
                "type": "integer",
            }

        case builtins.float:
            specification = {
                "type": "number",
            }

        case builtins.bool:
            specification = {
                "type": "boolean",
            }

        case types.NoneType:
            specification = {
                "type": "null",
            }

        case types.UnionType:
            alternatives: list[ParameterSpecification] = []
            for arg in resolved_args:
                alternative_specification: ParameterSpecification | Missing = (
                    parameter_specification(
                        annotation=arg,
                        description=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )
                )

                if not_missing(alternative_specification):
                    alternatives.append(alternative_specification)

                else:
                    # when at least one element can't be represented in specification
                    # then the whole thing can't be represented in specification
                    return MISSING

            specification = {"oneOf": alternatives}

        case builtins.tuple:  # pyright: ignore[reportUnknownMemberType]
            match resolved_args:
                case [tuple_type_annotation, builtins.Ellipsis]:
                    element_specification: ParameterSpecification | Missing = (
                        parameter_specification(
                            annotation=tuple_type_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        )
                    )

                    if not_missing(element_specification):
                        specification = {
                            "type": "array",
                            "items": element_specification,
                        }

                    else:
                        # when at least one element can't be represented in specification
                        # then the whole thing can't be represented in specification
                        return MISSING

                case [*elements_annotations]:
                    elements: list[ParameterSpecification] = []
                    for arg in elements_annotations:
                        element_specification: ParameterSpecification | Missing = (
                            parameter_specification(
                                annotation=arg,
                                description=None,
                                globalns=globalns,
                                localns=localns,
                                recursion_guard=recursion_guard,
                            )
                        )

                        if not_missing(element_specification):
                            elements.append(element_specification)

                        else:
                            # when at least one element can't be represented in specification
                            # then the whole thing can't be represented in specification
                            return MISSING

                    specification = {
                        "type": "array",
                        "prefixItems": elements,
                    }

        case parametrized if hasattr(parametrized, "__PARAMETERS__"):
            nested_specification: ParameterSpecification | Missing = (
                parametrized.__PARAMETERS_SPECIFICATION__
            )

            if not_missing(nested_specification):
                specification = nested_specification

            else:
                # when at least one element can't be represented in specification
                # then the whole thing can't be represented in specification
                return MISSING

        case data_class if is_dataclass(data_class):
            required: list[str] = []
            annotations: dict[str, Any] = {}
            for field in dataclass_fields(data_class):
                annotations[field.name] = field.type
                if (
                    field.default is not DATACLASS_MISSING
                    or field.default_factory is not DATACLASS_MISSING
                ):
                    required.append(field.name)

            dataclass_specification: ParameterSpecification | Missing = _annotations_specification(
                annotations,
                required=required,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset({*recursion_guard, data_class}),
            )

            if not_missing(dataclass_specification):
                specification = dataclass_specification

            else:
                # when at least one element can't be represented in specification
                # then the whole thing can't be represented in specification
                return MISSING

        case typed_dict if typing.is_typeddict(typed_dict):
            dict_specification: ParameterSpecification | Missing = _annotations_specification(
                typed_dict.__annotations__,
                required=typed_dict.__required_keys__,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset({*recursion_guard, typed_dict}),
            )

            if not_missing(dict_specification):
                specification = dict_specification

            else:
                # when at least one element can't be represented in specification
                # then the whole thing can't be represented in specification
                return MISSING

        case builtins.dict | collections_abc.Mapping:  # pyright: ignore[reportUnknownMemberType]
            match resolved_args:
                case (builtins.str, element_annotation):
                    element_specification: ParameterSpecification | Missing = (
                        parameter_specification(
                            annotation=element_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        )
                    )

                    if not_missing(element_specification):
                        specification = {
                            "type": "object",
                            "additionalProperties": element_specification,
                        }

                    else:
                        # when at least one element can't be represented in specification
                        # then the whole thing can't be represented in specification
                        return MISSING

                case _:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported dict type annotation", annotation)

        case builtins.set | collections_abc.Set:
            match resolved_args:
                case [list_type_annotation]:
                    element_specification: ParameterSpecification | Missing = (
                        parameter_specification(
                            annotation=list_type_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        )
                    )

                    if not_missing(element_specification):
                        specification = {
                            "type": "array",
                            "items": element_specification,
                        }

                    else:
                        # when at least one element can't be represented in specification
                        # then the whole thing can't be represented in specification
                        return MISSING

                case _:
                    raise TypeError("Unsupported set type annotation: %s", annotation)

        case builtins.list | collections_abc.Sequence:  # pyright: ignore[reportUnknownMemberType]
            match resolved_args:
                case [list_type_annotation]:
                    element_specification: ParameterSpecification | Missing = (
                        parameter_specification(
                            annotation=list_type_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        )
                    )

                    if not_missing(element_specification):
                        specification = {
                            "type": "array",
                            "items": element_specification,
                        }

                    else:
                        # when at least one element can't be represented in specification
                        # then the whole thing can't be represented in specification
                        return MISSING

                case _:
                    raise TypeError("Unsupported list type annotation: %s", annotation)

        case typing.Literal:
            if all(isinstance(option, str) for option in resolved_args):
                specification = {
                    "type": "string",
                    "enum": list(resolved_args),
                }

            elif all(isinstance(option, int) for option in resolved_args):
                specification = {
                    "type": "integer",
                    "enum": list(resolved_args),
                }

            elif all(isinstance(option, int | float) for option in resolved_args):
                specification = {
                    "type": "number",
                    "enum": list(resolved_args),
                }

            else:
                raise TypeError("Unsupported literal type annotation: %s", annotation)

        case enum_type if isinstance(enum_type, enum.EnumType):
            if isinstance(enum_type, enum.StrEnum):
                specification = {
                    "type": "string",
                    "enum": list(enum_type),
                }

            elif isinstance(enum_type, enum.IntEnum):
                specification = {
                    "type": "integer",
                    "enum": list(enum_type),
                }

            else:
                raise TypeError("Unsupported enum type annotation: %s", annotation)

        case datetime.datetime:
            specification = {
                "type": "string",
                "format": "date-time",
            }

        case uuid.UUID:
            specification = {
                "type": "string",
                "format": "uuid",
            }

        case other:
            return MISSING
            raise TypeError("Unsupported type annotation", other)

    if description := description:
        specification["description"] = description

    return specification


def _annotations_specification(
    annotations: dict[str, Any],
    /,
    required: Sequence[str],
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParametersSpecification | Missing:
    parameters: dict[str, ParameterSpecification] = {}
    for name, annotation in annotations.items():
        try:
            specification: ParameterSpecification | Missing = parameter_specification(
                annotation=annotation,
                description=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

            if not_missing(specification):
                parameters[name] = specification

            else:
                # when at least one element can't be represented in specification
                # then the whole thing can't be represented in specification
                return MISSING

        except Exception as exc:
            raise TypeError(f"Failed to extract parameter: {name}") from exc

    return {
        "type": "object",
        "properties": parameters,
        "required": list(required),
    }
