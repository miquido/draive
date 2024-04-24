import builtins
import datetime
import enum
import inspect
import types
import typing
import uuid
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import (
    Any,
    ForwardRef,
    Literal,
    NotRequired,
    Required,
    TypedDict,
    final,
    get_args,
    get_origin,
)

import typing_extensions

import draive.helpers.missing as draive_missing

__all__ = [
    "ParameterSpecification",
    "ParametersSpecification",
    "ToolSpecification",
    "parameter_specification",
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
class ParameterNumberSpecification(TypedDict, total=False):
    type: Required[Literal["number"]]
    description: NotRequired[str]


@final
class ParameterStringSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]
    description: NotRequired[str]
    format: NotRequired[
        Literal[
            "uuid",
            "date-time",
        ]
    ]


@final
class ParameterStringEnumSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]
    enum: Required[list[str]]
    description: NotRequired[str]


@final
class ParameterNumberEnumSpecification(TypedDict, total=False):
    type: Required[Literal["number"]]
    enum: Required[list[int | float]]
    description: NotRequired[str]


ParameterEnumSpecification = ParameterStringEnumSpecification | ParameterNumberEnumSpecification


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
class ParameterDictSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    additionalProperties: Required["ParameterSpecification"]
    description: NotRequired[str]
    required: NotRequired[list[str]]


@final
class ParameterObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[dict[str, "ParameterSpecification"]]
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
    | ParameterNumberSpecification
    | ParameterStringSpecification
    | ParameterEnumSpecification
    | ParameterUnionSpecification
    | ParameterArraySpecification
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


def _resolved_annotation(
    annotation: Any,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
) -> Any:
    if isinstance(annotation, str):
        return ForwardRef(annotation)._evaluate(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            globalns=globalns,
            localns=localns,
            recursive_guard=frozenset(),
        )
    elif isinstance(annotation, ForwardRef):
        return annotation._evaluate(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            globalns=globalns,
            localns=localns,
            recursive_guard=frozenset(),
        )
    else:
        return annotation


def parameter_specification(  # noqa: C901, PLR0912
    annotation: Any,
    description: str | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterSpecification:
    resolved_annotation: Any = _resolved_annotation(
        annotation,
        globalns=globalns,
        localns=localns,
    )
    if resolved_annotation in recursion_guard:
        # TODO: FIXME: recursive specification is not properly supported
        reference: ReferenceParameterSpecification = {"$ref": resolved_annotation.__qualname__}
        if description := description:
            reference["description"] = description
        return reference

    specification: ParameterSpecification
    match get_origin(resolved_annotation) or resolved_annotation:
        case builtins.str:
            specification = {
                "type": "string",
            }

        case builtins.int:
            specification = {
                "type": "number",
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

        case types.UnionType | typing.Union:
            specification = {
                "oneOf": [
                    parameter_specification(
                        annotation=arg,
                        description=description,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )
                    for arg in get_args(resolved_annotation)
                ]
            }

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(resolved_annotation):
                case (tuple_annotation,):
                    specification = {
                        "type": "array",
                        "items": parameter_specification(
                            annotation=tuple_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        ),
                    }

                case other:
                    specification = {
                        "type": "array",
                    }
        case (
            builtins.tuple  # pyright: ignore[reportUnknownMemberType]
            | typing.Tuple  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(resolved_annotation):
                case (tuple_annotation, builtins.Ellipsis | types.EllipsisType):
                    specification = {
                        "type": "array",
                        "items": parameter_specification(
                            annotation=tuple_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        ),
                    }

                # TODO: represent element type for finite tuples
                case other:
                    specification = {
                        "type": "array",
                    }

        case typing.Literal:
            options: tuple[Any, ...] = get_args(resolved_annotation)
            if all(isinstance(option, str) for option in options):
                specification = {
                    "type": "string",
                    "enum": list(options),
                }

            elif all(isinstance(option, int | float) for option in options):
                specification = {
                    "type": "number",
                    "enum": list(options),
                }

            else:
                raise TypeError("Unsupported literal type annotation", resolved_annotation)

        case enum_type if isinstance(enum_type, enum.EnumType):
            if isinstance(enum_type, enum.StrEnum):
                specification = {
                    "type": "string",
                    "enum": list(enum_type),
                }
            elif isinstance(enum_type, enum.IntEnum):
                specification = {
                    "type": "number",
                    "enum": list(enum_type),
                }
            else:
                raise TypeError("Unsupported enum type annotation", resolved_annotation)

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

        case parametrized if hasattr(parametrized, "__parameters_specification__"):
            specification = parametrized.__parameters_specification__

        case typed_dict if typing.is_typeddict(typed_dict) or typing_extensions.is_typeddict(
            typed_dict
        ):
            specification = _annotations_specification(
                typed_dict.__annotations__,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset({*recursion_guard, typed_dict}),
            )

        case (
            builtins.dict  # pyright: ignore[reportUnknownMemberType]
            | typing.Dict  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(resolved_annotation):
                case (builtins.str, element_annotation):
                    specification = {
                        "type": "object",
                        "additionalProperties": parameter_specification(
                            annotation=element_annotation,
                            description=None,
                            globalns=globalns,
                            localns=localns,
                            recursion_guard=recursion_guard,
                        ),
                    }

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported dict type annotation", other)

        case data_class if is_dataclass(data_class):
            specification = _function_specification(
                resolved_annotation.__init__,
                globalns=globalns,
                localns=localns,
                recursion_guard=frozenset({*recursion_guard, typed_dict}),
            )

        case typing.Annotated:
            match get_args(resolved_annotation):
                case [other, *_]:
                    return parameter_specification(
                        annotation=other,
                        description=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )

                case other:
                    raise TypeError("Unsupported annotated type annotation", other)

        case typing.Required | typing.NotRequired:
            match get_args(resolved_annotation):
                case [other, *_]:
                    return parameter_specification(
                        annotation=other,
                        description=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )

                case other:
                    raise TypeError("Unsupported required type annotation", other)

        case draive_missing.Missing:
            specification = {
                "type": "null",
            }

        case other:
            raise TypeError("Unsupported type annotation", other)

    if description := description:
        specification["description"] = description

    return specification


def _function_specification(
    function: Callable[..., Any],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParametersSpecification:
    parameters: dict[str, ParameterSpecification] = {}
    required: list[str] = []

    for parameter in inspect.signature(function).parameters.values():
        try:
            match (parameter.annotation, get_origin(parameter.annotation)):
                case (inspect._empty, _):  # pyright: ignore[reportPrivateUsage]
                    if parameter.name == "self":
                        continue  # skip object method "self" argument
                    else:
                        raise TypeError(
                            "Untyped argument %s",
                            parameter.name,
                        )

                case (annotation, _):
                    parameters[parameter.name] = parameter_specification(
                        annotation=annotation,
                        description=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                    )
                    if parameter.default is inspect._empty:  # pyright: ignore[reportPrivateUsage]
                        required.append(parameter.name)

        except Exception as exc:
            raise TypeError("Failed to extract parameter", parameter.name) from exc

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


def _annotations_specification(
    annotations: dict[str, Any],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParametersSpecification:
    parameters: dict[str, ParameterSpecification] = {}
    required: list[str] = []

    for name, annotation in annotations.items():
        try:
            origin: type[Any] = get_origin(annotation)
            parameters[name] = parameter_specification(
                annotation=annotation,
                description=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )
            # assuming total=True or explicitly annotated
            if origin != typing.NotRequired:
                required.append(name)

        except Exception as exc:
            raise TypeError("Failed to extract parameter", name) from exc

    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }
