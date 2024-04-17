import builtins
import datetime
import enum
import inspect
import types
import typing
import uuid
from collections import abc as collections_abc
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass
from typing import Any, ForwardRef, Protocol, TypeVar, get_args, get_origin

import typing_extensions

import draive.helpers.missing as draive_missing

__all__ = [
    "parameter_validator",
]

ValueVerifier = Callable[[Any], None]
ValueValidator = Callable[[Any], Any]


def _none_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if value is None:
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if value is None:
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _str_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, str):
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, str):
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _int_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, int):
                verify(value)
                return value
            elif isinstance(value, float) and value.is_integer():
                # auto-convert from float
                converted: int = int(value)
                verify(converted)
                return converted
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, int):
                return value
            elif isinstance(value, float) and value.is_integer():
                # auto-convert from float
                return int(value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _float_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, float):
                verify(value)
                return value
            elif isinstance(value, int):
                # auto-convert from int
                converted: float = float(value)
                verify(converted)
                return converted
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, float):
                return value
            elif isinstance(value, int):
                # auto-convert from int
                return float(value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _bool_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, bool):
                verify(value)
                return value
            elif isinstance(value, int):
                # auto-convert from int
                converted: bool = bool(value)
                verify(converted)
                return converted
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, bool):
                return value
            elif isinstance(value, int):
                # auto-convert from int
                return bool(value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _tuple_validator(
    options: tuple[Any, ...],
    verifier: ValueVerifier | None,
) -> ValueValidator:
    # TODO: validate tuple elements
    #  | types.EllipsisType
    # element_validators: list[Callable[[Any], Any]] = [
    #     parameter_validator(
    #         option,
    #         verifier=verifier,
    #         module=module,
    #     )
    #     for option in options
    # ]
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, tuple):
                verify(value)
                return value  # pyright: ignore[reportUnknownVariableType]
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, tuple):
                return value  # pyright: ignore[reportUnknownVariableType]
            else:
                raise TypeError("Invalid value", value)

    return validated


def _union_validator(
    alternatives: Iterable[Any],
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
    verifier: ValueVerifier | None,
) -> ValueValidator:
    validators: list[Callable[[Any], Any]] = [
        parameter_validator(
            alternative,
            verifier=verifier,
            globalns=globalns,
            localns=localns,
            recursion_guard=recursion_guard,
        )
        for alternative in alternatives
    ]
    if verify := verifier:

        def validated(value: Any) -> Any:
            for validator in validators:
                try:
                    validated: Any = validator(value)
                    verify(validated)
                    return validated
                except (ValueError, TypeError):
                    continue

            raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            for validator in validators:
                try:
                    return validator(value)
                except (ValueError, TypeError):
                    continue

            raise TypeError("Invalid value", value)

    return validated


def _list_validator(
    element_annotation: Any | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
    verifier: ValueVerifier | None,
) -> ValueValidator:
    validate_element: Callable[[Any], Any]
    if element_annotation is None:
        validate_element = lambda value: value  # noqa: E731

    else:
        validate_element = parameter_validator(
            element_annotation,
            globalns=globalns,
            localns=localns,
            recursion_guard=recursion_guard,
            verifier=None,
        )

    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, list):
                values_list: builtins.list[Any] = [
                    validate_element(element)
                    for element in value  # pyright: ignore[reportUnknownVariableType]
                ]
                verify(values_list)
                return values_list
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, list):
                return [
                    validate_element(element)
                    for element in value  # pyright: ignore[reportUnknownVariableType]
                ]
            else:
                raise TypeError("Invalid value", value)

    return validated


def _literal_validator(
    options: tuple[Any, ...],
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if value in options:
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if value in options:
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _str_enum_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            elif isinstance(value, str):
                enum_value: Any = annotation(value)
                verify(enum_value)
                return enum_value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            elif isinstance(value, str):
                return annotation(value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _int_enum_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            elif isinstance(value, int):
                enum_value: Any = annotation(value)
                verify(enum_value)
                return enum_value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            elif isinstance(value, int):
                return annotation(value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _enum_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _uuid_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, uuid.UUID):
                verify(value)
                return value
            elif isinstance(value, str):
                uuid_value: uuid.UUID = uuid.UUID(hex=value)
                verify(uuid_value)
                return uuid_value
            elif isinstance(value, bytes):
                uuid_value: uuid.UUID = uuid.UUID(bytes=value)
                verify(uuid_value)
                return uuid_value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, uuid.UUID):
                return value
            elif isinstance(value, str):
                uuid_value: uuid.UUID = uuid.UUID(hex=value)
                return uuid_value
            elif isinstance(value, bytes):
                uuid_value: uuid.UUID = uuid.UUID(bytes=value)
                return uuid_value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _datetime_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, datetime.datetime):
                verify(value)
                return value
            elif isinstance(value, str):
                datetime_value: datetime.datetime = datetime.datetime.fromisoformat(value)
                verify(datetime_value)
                return datetime_value
            elif isinstance(value, float | int):
                datetime_value: datetime.datetime = datetime.datetime.fromtimestamp(
                    value,
                    datetime.UTC,
                )
                verify(datetime_value)
                return datetime_value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, datetime.datetime):
                return value
            elif isinstance(value, str):
                return datetime.datetime.fromisoformat(value)
            elif isinstance(value, float | int):
                return datetime.datetime.fromtimestamp(
                    value,
                    datetime.UTC,
                )
            else:
                raise TypeError("Invalid value", value)

    return validated


def _callable_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    # TODO: validate callable signature
    if verify := verifier:

        def validated(value: Any) -> Any:
            if callable(value) or inspect.isfunction(value):
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if callable(value) or inspect.isfunction(value):
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _protocol_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _dataclass_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    # TODO: validate dataclass internals (i.e. nested dataclass)
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            elif isinstance(value, dict):
                validated: Any = annotation(**value)
                verify(validated)
                return validated
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            elif isinstance(value, dict):
                validated: Any = annotation(**value)
                return validated
            else:
                raise TypeError("Invalid value", value)

    return validated


def _dict_validator(
    element_annotation: Any,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
    verifier: ValueVerifier | None,
) -> ValueValidator:
    validate_element: Callable[[Any], Any] = parameter_validator(
        element_annotation,
        globalns=globalns,
        localns=localns,
        recursion_guard=recursion_guard,
        verifier=None,
    )

    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, dict):
                values_dict: builtins.dict[str, Any] = {
                    key: validate_element(element)
                    for key, element in value.items()  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(key, str)
                }
                verify(values_dict)
                return values_dict
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: validate_element(element)
                    for key, element in value.items()  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(key, str)
                }
            else:
                raise TypeError("Invalid value", value)

    return validated


def _typed_dict_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    # TODO: validate typed dict internals (i.e. nested typed dicts)
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, dict):
                typed_dict: Any = annotation(**value)
                verify(typed_dict)
                return typed_dict
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, dict):
                return annotation(**value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _parametrized_validator(
    annotation: Any,
    validator: Callable[..., Any],
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            elif isinstance(value, dict):
                parametrized: Any = validator(**value)
                verify(parametrized)
                return parametrized
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            elif isinstance(value, dict):
                return validator(**value)
            else:
                raise TypeError("Invalid value", value)

    return validated


def _class_instance_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, annotation):
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _type_match_validator(
    *annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            if type(value) in annotation:
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if type(value) in annotation:
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


def _any_validator(
    verifier: ValueVerifier | None,
) -> ValueValidator:
    if verify := verifier:

        def validated(value: Any) -> Any:
            verify(value)
            return value
    else:

        def validated(value: Any) -> Any:
            return value

    return validated


def _meta_type_validator(
    annotation: Any,
    verifier: ValueVerifier | None,
) -> ValueValidator:
    # TODO: FIXME: verify meta type
    if verify := verifier:

        def validated(value: Any) -> Any:
            if isinstance(value, type):
                verify(value)
                return value
            else:
                raise TypeError("Invalid value", value)
    else:

        def validated(value: Any) -> Any:
            if isinstance(value, type):
                return value
            else:
                raise TypeError("Invalid value", value)

    return validated


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


def parameter_validator[Value](  # noqa: PLR0911, C901
    parameter_annotation: Any,
    /,
    verifier: Callable[[Value], None] | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> Callable[[Any], Value]:
    annotation: Any = _resolved_annotation(
        parameter_annotation,
        globalns=globalns,
        localns=localns,
    )

    if annotation in recursion_guard:
        return annotation.validator

    match get_origin(annotation) or annotation:
        case builtins.str:
            return _str_validator(verifier=verifier)

        case builtins.int:
            return _int_validator(verifier=verifier)

        case builtins.float:
            return _float_validator(verifier=verifier)

        case builtins.bool:
            return _bool_validator(verifier=verifier)

        case types.NoneType | None:
            return _none_validator(verifier=verifier)

        case types.UnionType | typing.Union:
            return _union_validator(
                alternatives=get_args(annotation),
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
                verifier=verifier,
            )

        case (
            builtins.list  # pyright: ignore[reportUnknownMemberType]
            | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(annotation):
                case (element_annotation,):
                    return _list_validator(
                        element_annotation=element_annotation,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                        verifier=verifier,
                    )

                case other:
                    return _list_validator(
                        element_annotation=None,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                        verifier=verifier,
                    )

        case typing.Literal:
            return _literal_validator(
                options=get_args(annotation),
                verifier=verifier,
            )

        case enum_type if isinstance(enum_type, enum.EnumType):
            if isinstance(enum_type, enum.StrEnum):
                return _str_enum_validator(
                    annotation=enum_type,
                    verifier=verifier,
                )
            elif isinstance(enum_type, enum.IntEnum):
                return _int_enum_validator(
                    annotation=enum_type,
                    verifier=verifier,
                )
            else:
                return _enum_validator(
                    annotation=enum_type,
                    verifier=verifier,
                )

        case builtins.tuple:
            return _tuple_validator(
                options=get_args(annotation),
                verifier=verifier,
            )

        case uuid.UUID:
            return _uuid_validator(verifier=verifier)

        case datetime.datetime:
            return _datetime_validator(verifier=verifier)

        case parametrized if hasattr(parametrized, "__parameters_definition__"):
            return _parametrized_validator(
                annotation=parametrized,
                validator=parametrized.validated,
                verifier=verifier,
            )

        case typed_dict if typing.is_typeddict(typed_dict) or typing_extensions.is_typeddict(
            typed_dict
        ):
            return _typed_dict_validator(
                annotation=typed_dict,
                verifier=verifier,
            )

        case (
            builtins.dict  # pyright: ignore[reportUnknownMemberType]
            | typing.Dict  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
        ):
            match get_args(annotation):
                case (builtins.str, element_annotation):
                    return _dict_validator(
                        element_annotation=element_annotation,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                        verifier=verifier,
                    )

                case other:  # pyright: ignore[reportUnnecessaryComparison]
                    raise TypeError("Unsupported dict type annotation", other)

        case data_class if is_dataclass(data_class):
            return _dataclass_validator(
                annotation=data_class,
                verifier=verifier,
            )

        case protocol if isinstance(protocol, Protocol):
            return _protocol_validator(
                annotation=protocol,
                verifier=verifier,
            )

        case collections_abc.Callable:  # pyright: ignore[reportUnknownMemberType]
            return _callable_validator(verifier=verifier)

        case typing.Annotated:
            match get_args(annotation):
                case (other, *_):
                    return parameter_validator(
                        other,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                        verifier=verifier,
                    )

                case other:
                    raise TypeError("Unsupported annotated type", annotation)

        case draive_missing.Missing:

            def validated_missing(value: Any) -> Any:
                if draive_missing.is_missing(value):
                    return value

                else:
                    raise TypeError("Invalid value", value)

            return validated_missing

        case typing.Required | typing.NotRequired:
            match get_args(annotation):
                case (other,):
                    return parameter_validator(
                        other,
                        globalns=globalns,
                        localns=localns,
                        recursion_guard=recursion_guard,
                        verifier=verifier,
                    )

                case other:
                    raise TypeError("Unsupported type annotation", annotation)

        case type_var if isinstance(type_var, TypeVar):
            # TODO: FIXME: resolve against concrete contextual type if able
            if constraints := type_var.__constraints__:
                return _type_match_validator(
                    constraints,
                    verifier=verifier,
                )
            elif bound := type_var.__bound__:
                return _class_instance_validator(
                    bound,
                    verifier=verifier,
                )
            else:
                return _any_validator(verifier=verifier)

        case class_type if inspect.isclass(class_type):
            return _class_instance_validator(
                annotation=class_type,
                verifier=verifier,
            )

        case builtins.type:
            return _meta_type_validator(
                annotation=annotation,
                verifier=verifier,
            )

        case typing.Any:
            return _any_validator(verifier=verifier)

        case other:
            raise TypeError("Unsupported type annotation %s", other)
