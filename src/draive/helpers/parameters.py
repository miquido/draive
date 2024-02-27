import builtins
import types
import typing
from collections import abc as collections_abc
from collections.abc import Callable
from inspect import _empty, signature  # pyright: ignore[reportPrivateUsage]
from typing import (
    Any,
    Literal,
    NotRequired,
    Required,
    TypedDict,
    Unpack,
    final,
    get_args,
    get_origin,
)

__all__ = [
    "ParametersSpecification",
    "extract_parameters_specification",
]


@final
class ParameterNoneSpecification(TypedDict, total=False):
    type: Required[Literal["null"]]


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
class ParameterEnumSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]
    enum: Required[list[str]]
    description: NotRequired[str]


@final
class ParameterNestedBoolSpecification(TypedDict, total=False):
    type: Required[Literal["boolean"]]


@final
class ParameterNestedNumberSpecification(TypedDict, total=False):
    type: Required[Literal["number"]]


@final
class ParameterNestedStringSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]


@final
class ParameterNestedEnumSpecification(TypedDict, total=False):
    type: Required[Literal["string"]]
    enum: Required[list[str]]


@final
class ParameterNestedArraySpecification(TypedDict, total=False):
    type: Required[Literal["array"]]
    items: NotRequired["ParameterNestedSpecification"]


@final
class ParameterNestedObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[dict[str, "ParameterSpecification"]]


ParameterNestedSpecification = (
    ParameterNestedBoolSpecification
    | ParameterNestedNumberSpecification
    | ParameterNestedStringSpecification
    | ParameterNestedEnumSpecification
    | ParameterNestedArraySpecification
    | ParameterNestedObjectSpecification
)


@final
class ParameterUnionSpecification(TypedDict, total=False):
    oneOf: Required[list[ParameterNoneSpecification | ParameterNestedSpecification]]


@final
class ParameterArraySpecification(TypedDict, total=False):
    type: Required[Literal["array"]]
    items: NotRequired[ParameterNestedSpecification]
    description: NotRequired[str]


@final
class ParameterObjectSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[dict[str, "ParameterSpecification"]]
    description: NotRequired[str]


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

ParametersSpecification = dict[str, ParameterSpecification]


class ToolFunctionParametersSpecification(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Required[ParametersSpecification]


class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: NotRequired[str]
    parameters: Required[ToolFunctionParametersSpecification]


class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]


def _extract_type(
    __annotation: Any,
) -> type[Any]:
    match get_origin(__annotation):
        case typing.Annotated:
            raise TypeError("Unexpected Annotated")

        case types.UnionType | typing.Union:
            # allowing only "optional" union - type and None
            match get_args(__annotation):
                case (
                    (
                        optional_type,
                        types.NoneType,
                    )
                    | (
                        types.NoneType,
                        optional_type,
                    )
                ):
                    # TODO: it might be beneficial to express optional arguments
                    # as type unions in specification (which is json schema)
                    return optional_type

                case other:
                    raise TypeError("Unsupported union type annotation", other)

        case typing.Literal:
            return __annotation

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            | collections_abc.Sequence  # pyright: ignore[reportUnknownMemberType]
            | collections_abc.Iterable  # pyright: ignore[reportUnknownMemberType]
        ):
            return __annotation

        case None:  # no origin is a type itself
            return __annotation

        # TODO: add support for objects/Mapping
        case other:
            raise TypeError("Unsupported type annotation", other)


def _extract_argument(
    __annotation: Any,
) -> tuple[type[Any], str | None]:
    match get_origin(__annotation):
        case typing.Annotated:
            match get_args(__annotation):
                case [other, str() as description, *_tail]:
                    return (_extract_type(other), description)

                case [other, *_tail]:
                    return (_extract_type(other), None)

                case other:
                    raise TypeError("Unsupported annotated type annotation", other)

        case typing.Required | typing.NotRequired:
            match get_args(__annotation):
                case [other, *_tail]:
                    return _extract_argument(other)

                case other:
                    raise TypeError("Unsupported required type annotation", other)

        case other:
            return _extract_type(__annotation), None


def _parameter_specification(
    __argument: tuple[type[Any], str | None],
) -> ParameterSpecification:
    # allowing only selected types - available to use with AI
    specification: ParameterSpecification
    match get_origin(__argument[0]) or __argument[0]:
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
            specification = {
                "type": "string",
                "enum": list(get_args(__argument[0])),
            }

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            | collections_abc.Sequence  # pyright: ignore[reportUnknownMemberType]
            | collections_abc.Iterable  # pyright: ignore[reportUnknownMemberType]
        ):
            match get_args(__argument[0]):
                # allowing only str, int, float and bool lists or untyped
                case (builtins.str,):
                    specification = {
                        "type": "array",
                        "items": {"type": "string"},
                    }

                case (builtins.int,) | (builtins.float,):
                    specification = {
                        "type": "array",
                        "items": {"type": "number"},
                    }

                case (builtins.bool,):
                    specification = {
                        "type": "array",
                        "items": {"type": "boolean"},
                    }

                case ():  # pyright: ignore[reportUnnecessaryComparison] fallback to untyped list
                    specification = {
                        "type": "array",
                    }

                case other:
                    raise TypeError("Unsupported iterable type annotation", other)

        # TODO: add support for objects
        case other:
            raise TypeError("Unsupported type annotation", other)

    if description := __argument[1]:
        specification["description"] = description

    return specification


def extract_parameters_specification(
    function: Callable[..., Any],
    /,
) -> ParametersSpecification:
    parameters: ParametersSpecification = {}

    for parameter in signature(function).parameters.values():
        try:
            if parameter.annotation is _empty and parameter.name == "self":
                continue

            if get_origin(parameter.annotation) is Unpack:
                # this is a bit fragile - checking TypedDict seems to be hard?
                for unpacked in get_args(parameter.annotation):
                    for key, annotation in unpacked.__annotations__.items():
                        parameters[key] = _parameter_specification(_extract_argument(annotation))
            else:
                parameters[parameter.name] = _parameter_specification(
                    _extract_argument(parameter.annotation)
                )
        except Exception as exc:
            raise TypeError("Failed to extract parameter", parameter.name) from exc

    return parameters
