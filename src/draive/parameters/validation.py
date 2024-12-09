from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time
from enum import Enum
from types import MappingProxyType, NoneType, UnionType
from typing import Any, Literal, Union
from typing import Mapping as MappingType  # noqa: UP035
from typing import Sequence as SequenceType  # noqa: UP035
from uuid import UUID

from haiway import MISSING, Missing
from haiway.state.attributes import AttributeAnnotation

from draive.parameters.types import (
    ParameterValidationContext,
    ParameterValidator,
    ParameterVerifier,
)

__all__ = [
    "parameter_validator",
]


def parameter_validator(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None = None,
) -> ParameterValidator[Any]:
    if common := VALIDATORS.get(annotation.origin):
        return common(annotation, verifier)

    elif hasattr(annotation.origin, "validate"):
        # TODO: FIXME: add verifier?
        return annotation.origin.validate

    elif issubclass(annotation.origin, Enum):
        # TODO: FIXME: enums !!!
        return _prepare_validator_of_type(annotation, verifier)

    else:
        raise TypeError(f"Unsupported type annotation: {annotation}")


def _prepare_validator_of_any(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    def validator(
        value: Any,
        /,
        context: ParameterValidationContext,
    ) -> Any:
        return value  # any is always valid

    return validator


def _prepare_validator_of_none(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if value is None:
                verifier(value)
                return value

            else:
                raise TypeError(f"'{value}' is not matching expected type of 'None'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if value is None:
                return value

            else:
                raise TypeError(f"'{value}' is not matching expected type of 'None'")

    return validator


def _prepare_validator_of_missing(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if value is MISSING:
                verifier(value)
                return value

            else:
                raise TypeError(f"'{value}' is not matching expected type of 'Missing'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if value is MISSING:
                return value

            else:
                raise TypeError(f"'{value}' is not matching expected type of 'Missing'")

    return validator


def _prepare_validator_of_literal(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    elements: list[Any] = annotation.arguments
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if value in elements:
                verifier(value)
                return value

            else:
                raise ValueError(f"'{value}' is not matching expected values of '{formatted_type}'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if value in elements:
                return value

            else:
                raise ValueError(f"'{value}' is not matching expected values of '{formatted_type}'")

    return validator


def _prepare_validator_of_type(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    validated_type: type[Any] = annotation.origin
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case value if isinstance(value, validated_type):
                    verifier(value)
                    return value

                case _:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case value if isinstance(value, validated_type):
                    return value

                case _:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

    return validator


def _prepare_validator_of_sequence(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    element_validator: ParameterValidator[Any] = parameter_validator(
        annotation.arguments[0],
        verifier=None,
    )
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case [*elements]:
                    validated: Sequence[Any] = []
                    for idx, element in enumerate(elements):
                        with context.scope(f"[{idx}]"):
                            validated.append(element_validator(element, context))

                    validated = tuple(validated)
                    verifier(validated)
                    return validated

                case _:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case [*elements]:
                    validated: Sequence[Any] = []
                    for idx, element in enumerate(elements):
                        with context.scope(f"[{idx}]"):
                            validated.append(element_validator(element, context))

                    return tuple(validated)

                case _:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

    return validator


def _prepare_validator_of_mapping(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    key_validator: ParameterValidator[Any] = parameter_validator(annotation.arguments[0])
    value_validator: ParameterValidator[Any] = parameter_validator(annotation.arguments[1])
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case {**elements}:
                    validated: Mapping[Any, Any] = {}
                    for key, element in elements.items():
                        with context.scope(f"[{key}]"):
                            validated[key_validator(key, context)] = value_validator(
                                element, context
                            )

                    validated = MappingProxyType(validated)
                    verifier(validated)
                    return validated

                case _:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case {**elements}:
                    validated: Mapping[Any, Any] = {}
                    for key, element in elements.items():
                        with context.scope(f"[{key}]"):
                            validated[key_validator(key, context)] = value_validator(
                                element, context
                            )

                    return MappingProxyType(validated)

                case _:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

    return validator


def _prepare_validator_of_tuple(  # noqa: C901, PLR0915
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if annotation.arguments[-1].origin == Ellipsis:
        element_validator: ParameterValidator[Any] = parameter_validator(annotation.arguments[0])
        formatted_type: str = str(annotation)

        if verifier := verifier:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                match value:
                    case [*elements]:
                        validated: Sequence[Any] = []
                        for idx, element in enumerate(elements):
                            with context.scope(f"[{idx}]"):
                                validated.append(element_validator(element, context))

                        validated = tuple(validated)
                        verifier(validated)
                        return validated

                    case _:
                        raise TypeError(
                            f"'{value}' is not matching expected type of '{formatted_type}'"
                        )
        else:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                match value:
                    case [*elements]:
                        validated: Sequence[Any] = []
                        for idx, element in enumerate(elements):
                            with context.scope(f"[{idx}]"):
                                validated.append(element_validator(element, context))

                        return tuple(validated)

                    case _:
                        raise TypeError(
                            f"'{value}' is not matching expected type of '{formatted_type}'"
                        )

        return validator

    else:
        element_validators: list[ParameterValidator[Any]] = [
            parameter_validator(alternative) for alternative in annotation.arguments
        ]
        elements_count: int = len(element_validators)
        formatted_type: str = str(annotation)

        if verifier := verifier:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                match value:
                    case [*elements]:
                        if len(elements) != elements_count:
                            raise ValueError(
                                f"'{value}' is not matching expected type of '{formatted_type}'"
                            )

                        validated: Sequence[Any] = []
                        for idx, element in enumerate(elements):
                            with context.scope(f"[{idx}]"):
                                validated.append(element_validators[idx](element, context))

                        validated = tuple(validated)
                        verifier(validated)
                        return validated

                    case _:
                        raise TypeError(
                            f"'{value}' is not matching expected type of '{formatted_type}'"
                        )
        else:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                match value:
                    case [*elements]:
                        if len(elements) != elements_count:
                            raise ValueError(
                                f"'{value}' is not matching expected type of '{formatted_type}'"
                            )

                        validated: Sequence[Any] = []
                        for idx, element in enumerate(elements):
                            with context.scope(f"[{idx}]"):
                                validated.append(element_validators[idx](element, context))

                        return tuple(validated)

                    case _:
                        raise TypeError(
                            f"'{value}' is not matching expected type of '{formatted_type}'"
                        )

        return validator


def _prepare_validator_of_union(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    validators: list[ParameterValidator[Any]] = [
        parameter_validator(alternative) for alternative in annotation.arguments
    ]
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            errors: list[Exception] = []
            for validator in validators:
                try:
                    validated = validator(value, context)
                    verifier(validated)
                    return validated

                except Exception as exc:
                    errors.append(exc)

            raise ExceptionGroup(
                f"'{value}' is not matching expected type of '{formatted_type}'",
                errors,
            )
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            errors: list[Exception] = []
            for validator in validators:
                try:
                    return validator(value, context)

                except Exception as exc:
                    errors.append(exc)

            raise ExceptionGroup(
                f"'{value}' is not matching expected type of '{formatted_type}'",
                errors,
            )

    return validator


def _prepare_validator_of_float(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, float):
                verifier(value)
                return value

            elif isinstance(value, int):
                validated = float(value)
                verifier(validated)
                return validated

            raise TypeError(f"'{value}' is not matching expected type of 'float'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, float):
                return value

            elif isinstance(value, int):
                return float(value)

            raise TypeError(f"'{value}' is not matching expected type of 'float'")

    return validator


def _prepare_validator_of_bool(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, bool):
                verifier(value)
                return value

            elif isinstance(value, int):
                validated = bool(value != 0)
                verifier(validated)
                return validated

            elif isinstance(value, str):
                match value.lower():
                    case "true":
                        verifier(True)
                        return True

                    case "false":
                        verifier(False)
                        return False

                    case _:
                        pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'bool'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, bool):
                return value

            elif isinstance(value, int):
                return bool(value != 0)

            elif isinstance(value, str):
                match value.lower():
                    case "true":
                        return True

                    case "false":
                        return False

                    case _:
                        pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'bool'")

    return validator


def _prepare_validator_of_uuid(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, UUID):
                verifier(value)
                return value

            elif isinstance(value, str):
                validated = UUID(hex=value)
                verifier(validated)
                return validated

            raise TypeError(f"'{value}' is not matching expected type of 'UUID'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, UUID):
                return value

            elif isinstance(value, str):
                return UUID(hex=value)

            raise TypeError(f"'{value}' is not matching expected type of 'UUID'")

    return validator


def _prepare_validator_of_datetime(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, datetime):
                verifier(value)
                return value

            elif isinstance(value, str):
                validated = datetime.fromisoformat(value)
                verifier(validated)
                return validated

            elif isinstance(value, int | float):
                validated = datetime.fromtimestamp(
                    value,
                    UTC,
                )
                verifier(validated)
                return validated

            raise TypeError(f"'{value}' is not matching expected type of 'datetime'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, datetime):
                return value

            elif isinstance(value, str):
                return datetime.fromisoformat(value)

            elif isinstance(value, int | float):
                return datetime.fromtimestamp(
                    value,
                    UTC,
                )

            raise TypeError(f"'{value}' is not matching expected type of 'datetime'")

    return validator


def _prepare_validator_of_date(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, date):
                verifier(value)
                return value

            elif isinstance(value, str):
                validated = date.fromisoformat(value)
                verifier(validated)
                return validated

            elif isinstance(value, int | float):
                validated = date.fromtimestamp(value)
                verifier(validated)
                return validated

            raise TypeError(f"'{value}' is not matching expected type of 'datetime'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, date):
                return value

            elif isinstance(value, str):
                return date.fromisoformat(value)

            elif isinstance(value, int | float):
                return date.fromtimestamp(value)

            raise TypeError(f"'{value}' is not matching expected type of 'datetime'")

    return validator


def _prepare_validator_of_time(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerifier[Any] | None,
) -> ParameterValidator[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, datetime):
                verifier(value)
                return value

            elif isinstance(value, str):
                validated = datetime.fromisoformat(value)
                verifier(validated)
                return validated

            elif isinstance(value, int | float):
                validated = datetime.fromtimestamp(
                    value,
                    UTC,
                )
                verifier(validated)
                return validated

            raise TypeError(f"'{value}' is not matching expected type of 'datetime'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, datetime):
                return value

            elif isinstance(value, str):
                return datetime.fromisoformat(value)

            elif isinstance(value, int | float):
                return datetime.fromtimestamp(
                    value,
                    UTC,
                )

            raise TypeError(f"'{value}' is not matching expected type of 'datetime'")

    return validator


VALIDATORS: Mapping[
    Any, Callable[[AttributeAnnotation, ParameterVerifier[Any] | None], ParameterValidator[Any]]
] = {
    Any: _prepare_validator_of_any,
    NoneType: _prepare_validator_of_none,
    Missing: _prepare_validator_of_missing,
    type: _prepare_validator_of_type,
    bool: _prepare_validator_of_bool,
    int: _prepare_validator_of_type,
    float: _prepare_validator_of_float,
    str: _prepare_validator_of_type,
    tuple: _prepare_validator_of_tuple,
    Literal: _prepare_validator_of_literal,
    Sequence: _prepare_validator_of_sequence,
    SequenceType: _prepare_validator_of_sequence,
    Mapping: _prepare_validator_of_mapping,
    MappingType: _prepare_validator_of_mapping,
    UUID: _prepare_validator_of_uuid,
    date: _prepare_validator_of_date,
    datetime: _prepare_validator_of_datetime,
    time: _prepare_validator_of_time,
    Union: _prepare_validator_of_union,
    UnionType: _prepare_validator_of_union,
}
