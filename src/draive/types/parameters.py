import builtins
import inspect
import types
import typing
from collections import abc as collections_abc
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any, Literal, NotRequired, Required, TypedDict, cast, final, get_args, get_origin

__all__ = [
    "ParametersSpecification",
    "parameter_specification",
    "function_specification",
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
class ParameterObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[dict[str, "ParameterSpecification"]]
    description: NotRequired[str]
    required: NotRequired[list[str]]


ParameterSpecification = (
    ParameterNoneSpecification
    | ParameterBoolSpecification
    | ParameterNumberSpecification
    | ParameterStringSpecification
    | ParameterEnumSpecification
    | ParameterUnionSpecification
    | ParameterArraySpecification
    | ParameterObjectSpecification
)

ParametersSpecification = ParameterObjectSpecification


@final
class ToolFunctionParametersSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[ParametersSpecification]


@final
class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: NotRequired[str]
    parameters: Required[ToolFunctionParametersSpecification]
    required: NotRequired[list[str]]


@final
class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]


def parameter_specification(
    annotation: Any,
    origin: Any | None,
    description: str | None,
) -> ParameterSpecification:
    # allowing only selected types - available to use with AI
    specification: ParameterSpecification
    match origin or annotation:
        case types.NoneType:
            specification = {
                "type": "null",
            }

        case builtins.str:
            specification = {
                "type": "string",
            }

        case builtins.int | builtins.float:
            specification = {
                "type": "number",
            }

        case builtins.bool:
            specification = {
                "type": "boolean",
            }

        case typing.Literal:
            options: tuple[Any, ...] = get_args(annotation)
            if all(isinstance(option, str) for option in options):
                specification = {
                    "type": "string",
                    "enum": list(get_args(annotation)),
                }

            elif all(isinstance(option, int | float) for option in options):
                specification = {
                    "type": "number",
                    "enum": list(get_args(annotation)),
                }

            else:
                raise TypeError("Unsupported literal type annotation", annotation)

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            | collections_abc.Sequence  # pyright: ignore[reportUnknownMemberType]
            | collections_abc.Iterable  # pyright: ignore[reportUnknownMemberType]
        ):
            match get_args(annotation):
                case (list_annotation,):
                    specification = {
                        "type": "array",
                        "items": parameter_specification(
                            annotation=list_annotation,
                            origin=get_origin(list_annotation),
                            description=None,
                        ),
                    }

                case ():  # pyright: ignore[reportUnnecessaryComparison] fallback to untyped list
                    specification = {
                        "type": "array",
                    }

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported iterable type annotation", other)

        case typing.Annotated:
            match get_args(annotation):
                case [other, str() as other_description, *_]:
                    return parameter_specification(
                        annotation=other,
                        origin=get_origin(other),
                        description=other_description or description,
                    )

                case [other, *_]:
                    return parameter_specification(
                        annotation=other,
                        origin=get_origin(other),
                        description=None,
                    )

                case other:
                    raise TypeError("Unsupported annotated type annotation", other)

        case typing.Required | typing.NotRequired:
            match get_args(annotation):
                case [other, *_]:
                    return parameter_specification(
                        annotation=other,
                        origin=get_origin(other),
                        description=None,
                    )

                case other:
                    raise TypeError("Unsupported required type annotation", other)

        case types.UnionType | typing.Union:
            specification = {
                "oneOf": [
                    parameter_specification(
                        annotation=arg,
                        origin=get_origin(arg),
                        description=description,
                    )
                    for arg in get_args(annotation)
                ]
            }

        case other:
            if hasattr(other, "specification") and isinstance(other.specification, dict):
                specification = cast(ParameterSpecification, other.specification)  # pyright: ignore[reportUnknownMemberType]

            elif is_dataclass(other):
                specification = function_specification(annotation.__init__)

            # TODO: add support for typed dicts
            else:
                raise TypeError("Unsupported type annotation", other)

    if description := description:
        specification["description"] = description

    return specification


def function_specification(
    function: Callable[..., Any],
    /,
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

                case (annotation, typing.Unpack):
                    # this is a bit fragile - checking TypedDict seems to be hard?
                    for unpacked in get_args(annotation):
                        for key, annotation in unpacked.__annotations__.items():
                            parameters[key] = parameter_specification(
                                annotation=annotation,
                                origin=get_origin(annotation),
                                description=None,
                            )

                case (annotation, origin):
                    parameters[parameter.name] = parameter_specification(
                        annotation=annotation,
                        origin=origin,
                        description=None,
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
