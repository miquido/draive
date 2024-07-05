import builtins
import datetime
import enum
import types
import typing
import uuid
from collections import abc as collections_abc
from collections.abc import Callable, Sequence
from dataclasses import is_dataclass
from typing import Any

import draive.utils as draive_missing
from draive.parameters.annotations import resolve_annotation
from draive.parameters.errors import ParameterValidationContext, ParameterValidationError

__all__ = [
    "as_validator",
    "parameter_validator",
    "ParameterValidator",
    "ParameterVerifier",
]


type ParameterValidator[Value] = Callable[[Any, ParameterValidationContext], Value]
type ParameterVerifier[Value] = Callable[[Value], None]


def as_validator[Value](
    function: Callable[[Any], Value],
) -> ParameterValidator[Value]:
    def validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Value:
        try:
            return function(value)

        except Exception as exc:
            raise ParameterValidationError.invalid(
                exception=exc,
                context=context,
            ) from exc

    return validator


def _none_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case None:
            return None

        case _:
            raise ParameterValidationError.invalid_type(
                expected=types.NoneType,
                received=value,
                context=context,
            )


def _missing_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case missing if isinstance(missing, draive_missing.Missing):
            return missing

        case None:
            return draive_missing.MISSING

        case _:
            raise ParameterValidationError.invalid_type(
                expected=draive_missing.Missing,
                received=value,
                context=context,
            )


def _any_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    return value  # any is always valid


def _str_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case str() as value:
            return value

        case _:
            raise ParameterValidationError.invalid_type(
                expected=str,
                received=value,
                context=context,
            )


def _int_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case int() as value:
            return value

        case _:
            raise ParameterValidationError.invalid_type(
                expected=int,
                received=value,
                context=context,
            )


def _float_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case float() as value:
            return value

        case int() as convertible:
            return float(convertible)

        case _:
            raise ParameterValidationError.invalid_type(
                expected=float,
                received=value,
                context=context,
            )


def _bool_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case bool() as value:
            return value

        case int() as convertible:
            return bool(convertible != 0)

        case str() as convertible if convertible.lower() == "true":
            return True

        case str() as convertible if convertible.lower() == "false":
            return False

        case _:
            raise ParameterValidationError.invalid_type(
                expected=bool,
                received=value,
                context=context,
            )


def _prepare_tuple_validator(
    elements_annotation: tuple[Any, ...],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Any]:
    match elements_annotation:
        case [element, builtins.Ellipsis]:
            element_validator: ParameterValidator[Any] = parameter_validator(
                element,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

            def variable_tuple_validator(
                value: Any,
                context: ParameterValidationContext,
            ) -> Any:
                match value:
                    case [*values] as value:
                        return tuple[Any, ...](
                            element_validator(element, (*context, f"[{idx}]"))
                            for idx, element in enumerate(values)
                        )

                    case []:
                        return tuple[Any, ...]()

                    case _:
                        raise ParameterValidationError.invalid_type(
                            expected=tuple,
                            received=value,
                            context=context,
                        )

            return variable_tuple_validator

        case [*fixed]:
            element_validators: list[ParameterValidator[Any]] = [
                parameter_validator(
                    element,
                    verifier=None,
                    globalns=globalns,
                    localns=localns,
                    recursion_guard=recursion_guard,
                )
                for element in fixed
            ]

            def fixed_tuple_validator(
                value: Any,
                context: ParameterValidationContext,
            ) -> Any:
                match value:
                    case [*values] as value if len(values) == len(element_validators):
                        return tuple[Any, ...](
                            validator(element, (*context, f"[{idx}]"))
                            for (idx, element), validator in zip(
                                enumerate(values),
                                element_validators,
                                strict=True,
                            )
                        )

                    case _:
                        raise ParameterValidationError.invalid_type(
                            expected=tuple,
                            received=value,
                            context=context,
                        )

            return fixed_tuple_validator


def _prepare_list_validator(
    elements_annotation: tuple[Any, ...],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Any]:
    element_validator: ParameterValidator[Any]
    match elements_annotation:
        case [element]:
            element_validator = parameter_validator(
                element,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case other:
            raise TypeError(f"Unsupported annotation - list[{other}]")

    def list_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case [*values] as value:
                return list[Any](
                    element_validator(element, (*context, f"[{idx}]"))
                    for idx, element in enumerate(values)
                )

            case []:
                return list[Any]()

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=list,
                    received=value,
                    context=context,
                )

    return list_validator


def _prepare_set_validator(
    elements_annotation: tuple[Any, ...],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Any]:
    element_validator: ParameterValidator[Any]
    match elements_annotation:
        case [element]:
            element_validator = parameter_validator(
                element,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case other:
            raise TypeError(f"Unsupported annotation - set[{other}]")

    def set_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case [*values] as value:
                return set[Any](
                    element_validator(element, (*context, f"[{idx}]"))
                    for idx, element in enumerate(values)
                )

            case []:
                return set[Any]()

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=list,
                    received=value,
                    context=context,
                )

    return set_validator


def _prepare_union_validator(
    elements_annotation: tuple[Any, ...],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Any]:
    alternatives_validators: list[ParameterValidator[Any]] = [
        parameter_validator(
            alternative,
            verifier=None,
            globalns=globalns,
            localns=localns,
            recursion_guard=recursion_guard,
        )
        for alternative in elements_annotation
    ]

    def union_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        for validator in alternatives_validators:
            try:
                return validator(value, context)

            except ParameterValidationError:
                continue

        raise ParameterValidationError.invalid_type(
            expected=types.UnionType,
            received=value,
            context=context,
        )

    return union_validator


def _prepare_str_enum_validator(
    values: Sequence[str],
    /,
) -> ParameterValidator[Any]:
    def str_enum_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case str() as value:
                if value in values:
                    return value

                else:
                    raise ParameterValidationError.invalid_value(
                        expected=values,
                        received=value,
                        context=context,
                    )

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=str,
                    received=value,
                    context=context,
                )

    return str_enum_validator


def _prepare_int_enum_validator(
    values: Sequence[int],
    /,
) -> ParameterValidator[Any]:
    def int_enum_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case int() as value:
                if value in values:
                    return value

                else:
                    raise ParameterValidationError.invalid_value(
                        expected=values,
                        received=value,
                        context=context,
                    )

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=int,
                    received=value,
                    context=context,
                )

    return int_enum_validator


def _prepare_any_enum_validator(
    values: Sequence[Any],
    /,
) -> ParameterValidator[Any]:
    def enum_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        if value in values:
            return value

        else:
            raise ParameterValidationError.invalid_value(
                expected=values,
                received=value,
                context=context,
            )

    return enum_validator


def _uuid_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case str() as value:
            try:
                return uuid.UUID(hex=value)

            except Exception as exc:
                raise ParameterValidationError.invalid_value(
                    expected=uuid.UUID,
                    received=value,
                    context=context,
                ) from exc

        case uuid.UUID() as value:
            return value

        case bytes() as value:
            try:
                return uuid.UUID(bytes=value)

            except Exception as exc:
                raise ParameterValidationError.invalid_value(
                    expected=uuid.UUID,
                    received=value,
                    context=context,
                ) from exc

        case _:
            raise ParameterValidationError.invalid_type(
                expected=uuid.UUID,
                received=value,
                context=context,
            )


def _datetime_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        case str() as value:
            try:
                return datetime.datetime.fromisoformat(value)

            except Exception as exc:
                raise ParameterValidationError.invalid_value(
                    expected=datetime.datetime,
                    received=value,
                    context=context,
                ) from exc

        case datetime.datetime() as value:
            return value

        case int() as value:
            try:
                return datetime.datetime.fromtimestamp(
                    value,
                    datetime.UTC,
                )

            except Exception as exc:
                raise ParameterValidationError.invalid_value(
                    expected=datetime.datetime,
                    received=value,
                    context=context,
                ) from exc

        case float() as value:
            try:
                return datetime.datetime.fromtimestamp(
                    value,
                    datetime.UTC,
                )

            except Exception as exc:
                raise ParameterValidationError.invalid_value(
                    expected=datetime.datetime,
                    received=value,
                    context=context,
                ) from exc

        case _:
            raise ParameterValidationError.invalid_type(
                expected=datetime.datetime,
                received=value,
                context=context,
            )


def _callable_validator(
    value: Any,
    context: ParameterValidationContext,
) -> Any:
    match value:
        # TODO: verify signature eventually
        case value if callable(value):
            return value

        case _:
            raise ParameterValidationError.invalid_type(
                expected=collections_abc.Callable,
                received=value,
                context=context,
            )


def _prepare_type_validator(
    validated_type: type[Any],
    /,
) -> ParameterValidator[Any]:
    def type_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case value if isinstance(value, validated_type):
                return value

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=validated_type,
                    received=value,
                    context=context,
                )

    return type_validator


def _prepare_meta_type_validator(
    validated_type: type[Any],
    /,
) -> ParameterValidator[Any]:
    def type_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case type() as type_value:
                if type_value == validated_type:
                    return type_value

                else:
                    raise ParameterValidationError.invalid_value(
                        expected=validated_type,
                        received=type_value,
                        context=context,
                    )

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=type,
                    received=value,
                    context=context,
                )

    return type_validator


def _prepare_dataclass_validator(
    dataclass_type: type[Any],
    /,
) -> ParameterValidator[Any]:
    def dataclass_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case value if isinstance(value, dataclass_type):
                return value

            case {**parameters}:
                # TODO: nested validation?
                try:
                    return dataclass_type(**parameters)

                except Exception as exc:
                    raise ParameterValidationError.invalid_value(
                        expected=dataclass_type,
                        received=value,
                        context=context,
                    ) from exc

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=dataclass_type,
                    received=value,
                    context=context,
                )

    return dataclass_validator


def _prepare_dict_validator(
    items_annotation: tuple[Any, ...],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Any]:
    key_validator: ParameterValidator[Any]
    value_validator: ParameterValidator[Any]
    match items_annotation:
        case [key, value]:
            key_validator = parameter_validator(
                key,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )
            value_validator = parameter_validator(
                value,
                verifier=None,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case other:
            raise TypeError(f"Unsupported annotation - dict[{other}]")

    def dict_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case {**values}:
                return {
                    key_validator(key, context): value_validator(value, (*context, f"[{key}]"))
                    for key, value in values.items()
                }

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=dict,
                    received=value,
                    context=context,
                )

    return dict_validator


def _prepare_typed_dict_validator(
    typed_dict_annotation: type[Any],
    /,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Any]:
    required_fields: set[str] = typed_dict_annotation.__required_keys__
    field_validators: dict[str, ParameterValidator[Any]] = {
        name: parameter_validator(
            annotation,
            verifier=None,
            globalns=globalns,
            localns=localns,
            recursion_guard=recursion_guard,
        )
        for name, annotation in typed_dict_annotation.__annotations__.items()
    }

    def typed_dict_validator(
        value: Any,
        context: ParameterValidationContext,
    ) -> Any:
        match value:
            case {**values}:
                result: dict[str, Any] = {}
                for key, item in values.items():
                    match key:
                        case str() as name if name in field_validators:
                            result[key] = field_validators[name](item, (*context, f"[{key}]"))

                        case other:
                            raise ParameterValidationError.invalid_type(
                                expected=str,
                                received=other,
                                context=context,
                            )

                missing_values: set[str] = required_fields.difference(result.keys())
                if missing_values:
                    raise ParameterValidationError.missing(context=context)

                return result

            case _:
                raise ParameterValidationError.invalid_type(
                    expected=dict,
                    received=value,
                    context=context,
                )

    return typed_dict_validator


def _verified[Value](
    validator: ParameterValidator[Value],
    /,
    *,
    verifier: ParameterVerifier[Value],
) -> ParameterValidator[Value]:
    def wrapped(
        value: Any,
        context: ParameterValidationContext,
    ) -> Value:
        validated: Value = validator(value, context)

        try:
            verifier(validated)

        except ParameterValidationError as exc:
            raise exc

        except Exception as exc:
            raise ParameterValidationError.invalid(
                exception=exc,
                context=context,
            ) from exc

        return validated

    return wrapped


def parameter_validator[Value](  # noqa: C901, PLR0915, PLR0912
    annotation: Any,
    /,
    verifier: Callable[[Value], None] | None,
    globalns: dict[str, Any] | None,
    localns: dict[str, Any] | None,
    recursion_guard: frozenset[type[Any]],
) -> ParameterValidator[Value]:
    resolved_origin, resolved_args = resolve_annotation(
        annotation,
        globalns=globalns,
        localns=localns,
    )

    if resolved_origin in recursion_guard:
        return resolved_origin.validator

    validator: ParameterValidator[Value]
    match resolved_origin:
        case builtins.str:
            validator = _str_validator

        case builtins.int:
            validator = _int_validator

        case builtins.float:
            validator = _float_validator

        case builtins.bool:
            validator = _bool_validator

        case types.NoneType:
            validator = _none_validator

        case types.UnionType:
            validator = _prepare_union_validator(
                resolved_args,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case builtins.tuple:  # pyright: ignore[reportUnknownMemberType]
            validator = _prepare_tuple_validator(
                resolved_args,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case parametrized if hasattr(parametrized, "__PARAMETERS__"):
            validator = parametrized.validator

        case data_class if is_dataclass(data_class):
            validator = _prepare_dataclass_validator(data_class)

        case typed_dict if typing.is_typeddict(typed_dict):
            validator = _prepare_typed_dict_validator(
                typed_dict,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case builtins.dict | collections_abc.Mapping:  # pyright: ignore[reportUnknownMemberType]
            validator = _prepare_dict_validator(
                resolved_args,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case builtins.set | collections_abc.Set:
            validator = _prepare_set_validator(
                resolved_args,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case builtins.list | collections_abc.Sequence:  # pyright: ignore[reportUnknownMemberType]
            validator = _prepare_list_validator(
                resolved_args,
                globalns=globalns,
                localns=localns,
                recursion_guard=recursion_guard,
            )

        case typing.Literal:
            if all(isinstance(option, str) for option in resolved_args):
                validator = _prepare_str_enum_validator(resolved_args)

            elif all(isinstance(option, int) for option in resolved_args):
                validator = _prepare_int_enum_validator(resolved_args)

            else:
                validator = _prepare_any_enum_validator(resolved_args)

        case enum_type if isinstance(enum_type, enum.EnumType):
            if isinstance(enum_type, enum.StrEnum):
                validator = _prepare_str_enum_validator(resolved_args)

            elif isinstance(enum_type, enum.IntEnum):
                validator = _prepare_int_enum_validator(resolved_args)

            else:
                raise TypeError(f"Unsupported enum type annotation: {annotation}")

        case datetime.datetime:
            validator = _datetime_validator

        case uuid.UUID:
            validator = _uuid_validator

        case collections_abc.Callable:  # pyright: ignore[reportUnknownMemberType, reportUnnecessaryComparison]
            validator = _callable_validator

        case draive_missing.Missing:
            validator = _missing_validator

        case builtins.type:
            validator = _prepare_meta_type_validator(annotation)

        case typing.Any:
            validator = _any_validator

        case type() as other_type:
            validator = _prepare_type_validator(other_type)

        case other:
            raise TypeError("Unsupported type annotation: %s", other)

    if verifier := verifier:
        return _verified(validator, verifier=verifier)

    else:
        return validator
