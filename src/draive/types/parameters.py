import builtins
import inspect
import types
import typing
from collections import abc as collections_abc
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import (
    Any,
    Literal,
    NotRequired,
    Required,
    TypedDict,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
)

import typing_extensions

from draive.types.dictionary import DictionaryRepresentable
from draive.types.missing import MissingValue

__all__ = [
    "ParameterDefinition",
    "ParametersSpecification",
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


_ParameterType_T = TypeVar("_ParameterType_T")


@final
class ParameterDefinition:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        alias: str | None,
        description: str | None,
        annotation: Any,
        default: Callable[[], _ParameterType_T] | _ParameterType_T | MissingValue,
        validator: Callable[[_ParameterType_T], None] | None,
    ) -> None:
        assert name != alias, "Alias can't be the same as name"  # nosec: B101
        self.name: str = name
        self.alias: str | None = alias
        self.description: str | None = description
        self.annotation: Any = annotation
        self.default: Callable[[], _ParameterType_T] | _ParameterType_T | MissingValue = default
        self.validator: Callable[[_ParameterType_T], _ParameterType_T] = _prepare_validator(
            annotation=annotation,
            additional=validator,
        )

        def frozen(
            __name: str,
            __value: Any,
        ) -> None:
            raise RuntimeError("ParameterDefinition can't be modified")

        self.__setattr__ = frozen

    def default_value(self) -> Any | MissingValue:
        if callable(self.default):
            return self.default()
        else:
            return self.default

    def validated_value(
        self,
        value: Any,
    ) -> Any:
        return self.validator(value)

    def specification(self) -> ParameterSpecification:
        return _parameter_specification(
            annotation=self.annotation,
            description=self.description,
            origin=get_origin(self.annotation),
        )


def _parameter_specification(  # noqa: C901
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
                        "items": _parameter_specification(
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
                    return _parameter_specification(
                        annotation=other,
                        origin=get_origin(other),
                        description=other_description or description,
                    )

                case [other, *_]:
                    return _parameter_specification(
                        annotation=other,
                        origin=get_origin(other),
                        description=None,
                    )

                case other:
                    raise TypeError("Unsupported annotated type annotation", other)

        case typing.Required | typing.NotRequired:
            match get_args(annotation):
                case [other, *_]:
                    return _parameter_specification(
                        annotation=other,
                        origin=get_origin(other),
                        description=None,
                    )

                case other:
                    raise TypeError("Unsupported required type annotation", other)

        case types.UnionType | typing.Union:
            specification = {
                "oneOf": [
                    _parameter_specification(
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
                specification = _function_specification(annotation.__init__)

            elif typing.is_typeddict(other) or typing_extensions.is_typeddict(other):
                specification = _annotations_specification(other.__annotations__)

            else:
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

                case (annotation, typing.Unpack):
                    # this is a bit fragile - checking TypedDict seems to be hard?
                    for unpacked in get_args(annotation):
                        for key, annotation in unpacked.__annotations__.items():
                            parameters[key] = _parameter_specification(
                                annotation=annotation,
                                origin=get_origin(annotation),
                                description=None,
                            )

                case (annotation, origin):
                    parameters[parameter.name] = _parameter_specification(
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
                origin=origin,
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


def _prepare_validator(  # noqa: C901, PLR0911, PLR0915
    annotation: Any,
    additional: Callable[[Any], None] | None,
) -> Callable[[Any], Any]:
    match get_origin(annotation) or annotation:
        case typing.Annotated:
            match get_args(annotation):
                case [annotated, *_]:
                    return _prepare_validator(
                        annotation=annotated,
                        additional=additional,
                    )
                case annotated:
                    raise TypeError("Unsupported annotated type", annotated)

        case typing.Literal:

            def validated(value: Any) -> Any:
                if value in get_args(annotation):
                    if validate := additional:
                        validate(value)
                    return value
                else:
                    raise TypeError("Invalid value", annotation, value)

            return validated

        case types.UnionType | typing.Union:
            validators: list[Callable[[Any], Any]] = [
                _prepare_validator(
                    annotation=alternative,
                    additional=additional,
                )
                for alternative in get_args(annotation)
            ]

            def validated(value: Any) -> Any:
                for validator in validators:
                    try:
                        return validator(value)
                    except (ValueError, TypeError):
                        continue

                raise TypeError("Invalid value", annotation, value)

            return validated

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            | collections_abc.Sequence  # pyright: ignore[reportUnknownMemberType]
            | collections_abc.Iterable  # pyright: ignore[reportUnknownMemberType]
        ):
            validate_element: Callable[[Any], Any]
            match get_args(annotation):
                case (list_annotation,):
                    validate_element = _prepare_validator(
                        annotation=list_annotation,
                        additional=None,
                    )

                case ():  # pyright: ignore[reportUnnecessaryComparison] fallback to untyped list
                    validate_element = lambda value: value  # noqa: E731

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported iterable type annotation", other)

            def validated(value: Any) -> Any:
                values_list: list[Any]
                if isinstance(value, collections_abc.Iterable):
                    values_list = [validate_element(element) for element in value]  # pyright: ignore[reportUnknownVariableType]
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(values_list)
                return values_list

            return validated

        case model_type if issubclass(model_type, DictionaryRepresentable):

            def validated(value: Any) -> Any:
                model: DictionaryRepresentable
                if isinstance(value, dict):
                    model = model_type.from_dict(values=cast(dict[str, Any], value))
                elif isinstance(value, model_type):
                    model = value
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(model)
                return model

            return validated

        case typed_dict_type if typing.is_typeddict(
            typed_dict_type
        ) or typing_extensions.is_typeddict(typed_dict_type):

            def validated(value: Any) -> Any:
                typed_dict: dict[Any, Any]
                if isinstance(value, dict):
                    typed_dict = typed_dict_type(**value)
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(typed_dict)
                return typed_dict

            return validated

        case typed_dict_type if typing.is_typeddict(
            typed_dict_type
        ) or typing_extensions.is_typeddict(typed_dict_type):

            def validated(value: Any) -> Any:
                typed_dict: dict[Any, Any]
                if isinstance(value, dict):
                    typed_dict = typed_dict_type(**value)
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(typed_dict)
                return typed_dict

            return validated

        case other_type:

            def validated(value: Any) -> Any:
                if isinstance(value, other_type):
                    if validate := additional:
                        validate(value)
                    return value
                elif isinstance(value, float) and other_type == int:
                    # auto convert float to int - json does not distinguish those
                    converted_int: int = int(value)
                    if validate := additional:
                        validate(converted_int)
                    return converted_int
                elif isinstance(value, int) and other_type == float:
                    # auto convert int to float - json does not distinguish those
                    converted_float: float = float(value)
                    if validate := additional:
                        validate(converted_float)
                    return converted_float
                # TODO: validate function/callable values
                elif callable(value):
                    if validate := additional:
                        validate(value)
                    return value
                else:
                    raise TypeError("Invalid value", annotation, value)

            return validated
