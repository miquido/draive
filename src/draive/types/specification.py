import builtins
import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import (
    Any,
    Literal,
    NotRequired,
    ParamSpec,
    Required,
    TypedDict,
    TypeVar,
    final,
    get_args,
    get_origin,
)

import typing_extensions

from draive.types.parameters import (
    Function,
    ParametersDefinition,
    ParametrizedFunction,
    ParametrizedState,
)

__all__ = [
    "ParameterSpecification",
    "ParametersSpecification",
    "ParametrizedModel",
    "ToolSpecification",
    "ParametrizedTool",
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
)

ParametersSpecification = ParameterObjectSpecification


class ParametrizedModel(ParametrizedState):
    @classmethod
    def specification(cls) -> ParametersSpecification:
        if not hasattr(cls, "_specification"):
            definition: ParametersDefinition = cls.parameters()
            cls._specification: ParametersSpecification = {
                "type": "object",
                "properties": {
                    parameter.alias or parameter.name: _parameter_specification(
                        annotation=parameter.annotation,
                        description=parameter.description,
                    )
                    for parameter in definition.parameters
                },
                "required": definition.aliased_required,
            }

        return cls._specification


@final
class ToolFunctionSpecification(TypedDict, total=False):
    name: Required[str]
    description: Required[str]
    parameters: Required[ParametersSpecification]


@final
class ToolSpecification(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[ToolFunctionSpecification]


ToolArgs = ParamSpec(
    name="ToolArgs",
    # bound= - ideally it should be bound to allowed types, not implemented in python yet
)
ToolResult = TypeVar(
    name="ToolResult",
)


class ParametrizedTool(ParametrizedFunction[ToolArgs, ToolResult]):
    def __init__(
        self,
        /,
        name: str,
        *,
        function: Function[ToolArgs, ToolResult],
        description: str | None = None,
    ) -> None:
        super().__init__(function=function)
        self.name: str = name
        self.description: str | None = description
        self.specification: ToolSpecification = {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": {
                    "type": "object",
                    "properties": {
                        parameter.alias or parameter.name: _parameter_specification(
                            annotation=parameter.annotation,
                            description=parameter.description,
                        )
                        for parameter in self.parameters.parameters
                    },
                    "required": self.parameters.aliased_required,
                },
                "description": self.description or "",
            },
        }


def _parameter_specification(
    annotation: Any,
    description: str | None,
) -> ParameterSpecification:
    specification: ParameterSpecification
    match get_origin(annotation) or annotation:
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

        case types.UnionType | typing.Union:
            specification = {
                "oneOf": [
                    _parameter_specification(
                        annotation=arg,
                        description=description,
                    )
                    for arg in get_args(annotation)
                ]
            }

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(annotation):
                case (list_annotation,):
                    specification = {
                        "type": "array",
                        "items": _parameter_specification(
                            annotation=list_annotation,
                            description=None,
                        ),
                    }

                case ():  # pyright: ignore[reportUnnecessaryComparison] fallback to untyped list
                    specification = {
                        "type": "array",
                    }

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported list type annotation", other)

        case typing.Literal:
            options: tuple[Any, ...] = get_args(annotation)
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
                raise TypeError("Unsupported literal type annotation", annotation)

        case parametrized if issubclass(parametrized, ParametrizedModel):
            specification = parametrized.specification()

        case typed_dict if typing.is_typeddict(typed_dict) or typing_extensions.is_typeddict(
            typed_dict
        ):
            specification = _annotations_specification(typed_dict.__annotations__)

        case (
            builtins.dict  # pyright: ignore[reportUnknownMemberType]
            | typing.Dict  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(annotation):
                case (builtins.str, element_annotation):
                    specification = {
                        "type": "object",
                        "additionalProperties": _parameter_specification(
                            annotation=element_annotation,
                            description=None,
                        ),
                    }

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported dict type annotation", other)

        case data_class if is_dataclass(data_class):
            specification = _function_specification(annotation.__init__)

        case typing.Required | typing.NotRequired:
            match get_args(annotation):
                case [other, *_]:
                    return _parameter_specification(
                        annotation=other,
                        description=None,
                    )

                case other:
                    raise TypeError("Unsupported required type annotation", other)

        case typing.Annotated:
            match get_args(annotation):
                case [other, *_]:
                    return _parameter_specification(
                        annotation=other,
                        description=None,
                    )

                case other:
                    raise TypeError("Unsupported annotated type annotation", other)

        case other:
            raise TypeError("Unsupported type annotation", other)

    if description := description:
        specification["description"] = description

    return specification


def _function_specification(
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

                case (annotation, _):
                    parameters[parameter.name] = _parameter_specification(
                        annotation=annotation,
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


def _annotations_specification(
    annotations: dict[str, Any],
    /,
) -> ParametersSpecification:
    parameters: dict[str, ParameterSpecification] = {}
    required: list[str] = []

    for name, annotation in annotations.items():
        try:
            origin: type[Any] = get_origin(annotation)
            parameters[name] = _parameter_specification(
                annotation=annotation,
                description=None,
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
