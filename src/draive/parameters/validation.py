from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence, Set
from datetime import UTC, date, datetime, time
from enum import Enum, IntEnum, StrEnum
from types import EllipsisType, NoneType, UnionType
from typing import Any, Literal, Self, Union, is_typeddict
from uuid import UUID

from haiway import MISSING, Missing
from haiway.state.attributes import (
    AttributeAnnotation,
    _resolve_type_typeddict,
)

from draive.commons import Meta
from draive.commons.metadata import validated_meta_value
from draive.parameters.types import (
    ParameterValidation,
    ParameterValidationContext,
    ParameterVerification,
)

__all__ = ("ParameterValidator",)


class ParameterValidator[Type]:
    @classmethod
    def of_typed_dict[Dict](
        cls,
        typed_dict: type[Dict],
        /,
    ) -> ParameterValidation[Dict]:
        if not is_typeddict(typed_dict):
            raise ValueError("Type has to be a TypedDict")

        return cls.of(
            _resolve_type_typeddict(
                typed_dict,
                module=typed_dict.__module__,
                type_parameters={},  # TODO: type parameters?
                self_annotation=None,
                recursion_guard={},
            ),
            verifier=None,
            recursion_guard={},
        )

    @classmethod
    def of(
        cls,
        annotation: AttributeAnnotation,
        /,
        *,
        verifier: ParameterVerification[Any] | None,
        recursion_guard: MutableMapping[str, ParameterValidation[Any]],
    ) -> ParameterValidation[Any]:
        if isinstance(annotation.origin, NotImplementedError | RuntimeError):
            raise annotation.origin  # raise an error if origin was not properly resolved

        if recursive := recursion_guard.get(str(annotation)):
            return recursive

        validator: Self = cls(
            annotation,
            validation=MISSING,
        )
        recursion_guard[str(annotation)] = validator

        if isinstance(annotation.origin, NotImplementedError | RuntimeError):
            raise annotation.origin  # raise an error if origin was not properly resolved

        if common := VALIDATORS.get(annotation.origin):
            validator.validation = common(annotation, verifier, recursion_guard)

        elif hasattr(annotation.origin, "model_validator"):
            validator.validation = annotation.origin.model_validator(verifier=verifier)

        elif is_typeddict(annotation.origin):
            validator.validation = _prepare_validator_of_typed_dict(
                annotation, verifier, recursion_guard
            )

        elif issubclass(annotation.origin, StrEnum):
            # TODO: FIXME: str enums !!!
            validator.validation = _prepare_validator_of_enum(annotation, verifier, recursion_guard)

        elif issubclass(annotation.origin, IntEnum):
            # TODO: FIXME: int enums !!!
            validator.validation = _prepare_validator_of_enum(annotation, verifier, recursion_guard)

        elif issubclass(annotation.origin, Enum):
            validator.validation = _prepare_validator_of_enum(annotation, verifier, recursion_guard)

        else:
            raise TypeError(f"Unsupported type annotation: {annotation}")

        return validator

    def __init__(
        self,
        annotation: AttributeAnnotation,
        validation: ParameterValidation[Type] | Missing,
    ) -> None:
        self.annotation: AttributeAnnotation = annotation
        self.validation: ParameterValidation[Type] | Missing = validation

    def __call__(
        self,
        value: Any,
        /,
        *,
        context: ParameterValidationContext,
    ) -> Any:
        assert self.validation is not MISSING  # nosec: B101
        return self.validation(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            value,
            context=context,
        )


def _prepare_validator_of_any(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    elements: Sequence[Any] = annotation.arguments
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


def _prepare_validator_of_enum(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    elements: Sequence[Any] = list(annotation.origin)
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    element_validator: ParameterValidation[Any] = ParameterValidator.of(
        annotation.arguments[0],
        verifier=None,
        recursion_guard=recursion_guard,
    )
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Collection) and not isinstance(value, str | bytes):
                validated: Sequence[Any] = []
                for idx, element in enumerate(value):
                    with context.scope(f"[{idx}]"):
                        validated.append(element_validator(element, context=context))

                validated = tuple(validated)
                verifier(validated)
                return validated

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Collection) and not isinstance(value, str | bytes):
                validated: Sequence[Any] = []
                for idx, element in enumerate(value):
                    with context.scope(f"[{idx}]"):
                        validated.append(element_validator(element, context=context))

                return tuple(validated)

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    return validator


def _prepare_validator_of_mapping(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    key_validator: ParameterValidation[Any] = ParameterValidator.of(
        annotation.arguments[0],
        verifier=None,
        recursion_guard=recursion_guard,
    )
    value_validator: ParameterValidation[Any] = ParameterValidator.of(
        annotation.arguments[1],
        verifier=None,
        recursion_guard=recursion_guard,
    )
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Mapping):
                validated: MutableMapping[Any, Any] = {}
                for key, element in value.items():
                    with context.scope(f"[{key}]"):
                        validated[key_validator(key, context=context)] = value_validator(
                            element, context=context
                        )

                # TODO: FIXME: make sure dict is not mutable?
                # validated = MappingProxyType(validated)
                verifier(validated)
                return validated

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Mapping):
                validated: MutableMapping[Any, Any] = {}
                for key, element in value.items():
                    with context.scope(f"[{key}]"):
                        validated[key_validator(key, context=context)] = value_validator(
                            element, context=context
                        )

                # TODO: FIXME: make sure dict is not mutable?
                # validated = MappingProxyType(validated)
                return validated

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    return validator


def _prepare_validator_of_meta(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    formatted_type: str = str(annotation)

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Meta):
                return value

            elif isinstance(value, Mapping):
                validated: MutableMapping[Any, Any] = {}
                for key, element in value.items():
                    if not isinstance(key, str):
                        raise TypeError(f"'{key}' is not matching expected type of '{str}'")

                    with context.scope(f"[{key}]"):
                        validated[key] = validated_meta_value(element)

                meta: Meta = Meta(validated)
                verifier(meta)
                return meta

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Meta):
                return value

            elif isinstance(value, Mapping):
                validated: MutableMapping[Any, Any] = {}
                for key, element in value.items():
                    if not isinstance(key, str):
                        raise TypeError(f"'{key}' is not matching expected type of '{str}'")

                    with context.scope(f"[{key}]"):
                        validated[key] = validated_meta_value(element)

                return Meta(validated)

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    return validator


def _prepare_validator_of_tuple(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    if (
        annotation.arguments[-1].origin == Ellipsis
        or annotation.arguments[-1].origin == EllipsisType
    ):
        element_validator: ParameterValidation[Any] = ParameterValidator.of(
            annotation.arguments[0],
            verifier=None,
            recursion_guard=recursion_guard,
        )
        formatted_type: str = str(annotation)

        if verifier := verifier:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                if isinstance(value, Collection) and not isinstance(value, str | bytes):
                    validated: Sequence[Any] = []
                    for idx, element in enumerate(value):
                        with context.scope(f"[{idx}]"):
                            validated.append(element_validator(element, context=context))

                    validated = tuple(validated)
                    verifier(validated)
                    return validated

                else:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

        else:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                if isinstance(value, Collection) and not isinstance(value, str | bytes):
                    validated: Sequence[Any] = []
                    for idx, element in enumerate(value):
                        with context.scope(f"[{idx}]"):
                            validated.append(element_validator(element, context=context))

                    return tuple(validated)

                else:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

        return validator

    else:
        element_validators: list[ParameterValidation[Any]] = [
            ParameterValidator.of(
                alternative,
                verifier=None,
                recursion_guard=recursion_guard,
            )
            for alternative in annotation.arguments
        ]
        elements_count: int = len(element_validators)
        formatted_type: str = str(annotation)

        if verifier := verifier:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                if isinstance(value, Collection) and not isinstance(value, str | bytes):
                    if len(value) != elements_count:
                        raise ValueError(
                            f"'{value}' is not matching expected type of '{formatted_type}'"
                        )

                    validated: Sequence[Any] = []
                    for idx, element in enumerate(value):
                        with context.scope(f"[{idx}]"):
                            validated.append(element_validators[idx](element, context=context))

                    validated = tuple(validated)
                    verifier(validated)
                    return validated

                else:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

        else:

            def validator(
                value: Any,
                /,
                context: ParameterValidationContext,
            ) -> Any:
                if isinstance(value, Collection) and not isinstance(value, str | bytes):
                    if len(value) != elements_count:
                        raise ValueError(
                            f"'{value}' is not matching expected type of '{formatted_type}'"
                        )
                    validated: Sequence[Any] = []
                    for idx, element in enumerate(value):
                        with context.scope(f"[{idx}]"):
                            validated.append(element_validators[idx](element, context=context))

                    return tuple(validated)

                else:
                    raise TypeError(
                        f"'{value}' is not matching expected type of '{formatted_type}'"
                    )

        return validator


def _prepare_validator_of_union(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    validators: list[ParameterValidation[Any]] = [
        ParameterValidator.of(
            alternative,
            verifier=None,
            recursion_guard=recursion_guard,
        )
        for alternative in annotation.arguments
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
                    validated = validator(value, context=context)
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
                    return validator(value, context=context)

                except Exception as exc:
                    errors.append(exc)

            raise ExceptionGroup(
                f"'{value}' is not matching expected type of '{formatted_type}'",
                errors,
            )

    return validator


def _prepare_validator_of_float(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case float():
                    verifier(value)
                    return value

                case int():
                    validated = float(value)
                    verifier(validated)
                    return validated

                case str():
                    try:
                        validated = float(value)

                    except (ValueError, TypeError):
                        pass  # invalid

                    else:
                        verifier(validated)
                        return validated

                case _:
                    pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'float'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case float():
                    return value

                case int():
                    return float(value)

                case str():
                    try:
                        return float(value)

                    except (ValueError, TypeError):
                        pass  # invalid

                case _:
                    pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'float'")

    return validator


def _prepare_validator_of_int(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case int():
                    verifier(value)
                    return value

                case float():
                    validated = int(value)
                    if value == validated:
                        verifier(validated)
                        return int(validated)

                    # else invalid

                case str():
                    try:
                        validated = int(value)

                    except (ValueError, TypeError):
                        pass  # invalid

                    else:
                        verifier(validated)
                        return validated

                case _:
                    pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'int'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case int():
                    return value

                case float():
                    validated = int(value)
                    if value == validated:
                        return int(validated)

                    # else invalid

                case str():
                    try:
                        return int(value)

                    except (ValueError, TypeError):
                        pass  # invalid

                case _:
                    pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'int'")

    return validator


def _prepare_validator_of_bool(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case bool():
                    verifier(value)
                    return value

                case int():
                    validated = bool(value != 0)
                    verifier(validated)
                    return validated

                case str():
                    match value.lower():
                        case "true" | "t" | "1":
                            verifier(True)
                            return True

                        case "false" | "f" | "0":
                            verifier(False)
                            return False

                        case _:
                            pass  # invalid

                case _:
                    pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'bool'")
    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            match value:
                case bool():
                    return value

                case int():
                    validated = bool(value != 0)
                    return validated

                case str():
                    match value.lower():
                        case "true" | "t" | "1":
                            return True

                        case "false" | "f" | "0":
                            return False

                        case _:
                            pass  # invalid

                case _:
                    pass  # invalid

            raise TypeError(f"'{value}' is not matching expected type of 'bool'")

    return validator


def _prepare_validator_of_uuid(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
                try:
                    validated = UUID(value)

                except ValueError:
                    raise TypeError(f"'{value}' is not matching expected type of 'UUID'") from None

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
                try:
                    return UUID(value)

                except ValueError:
                    raise TypeError(f"'{value}' is not matching expected type of 'UUID'") from None

            raise TypeError(f"'{value}' is not matching expected type of 'UUID'")

    return validator


def _prepare_validator_of_datetime(
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
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


def _prepare_validator_of_typed_dict(  # noqa: C901
    annotation: AttributeAnnotation,
    /,
    verifier: ParameterVerification[Any] | None,
    recursion_guard: MutableMapping[str, ParameterValidation[Any]],
) -> ParameterValidation[Any]:
    def key_validator(
        value: Any,
    ) -> str:
        if isinstance(value, str):
            return value

        else:
            raise TypeError(f"'{value}' is not matching expected type of 'str'")

    formatted_type: str = str(annotation)
    values_validators: dict[str, ParameterValidation[Any]] = {
        key: ParameterValidator.of(
            element,
            verifier=None,
            recursion_guard=recursion_guard,
        )
        for key, element in annotation.extra["attributes"].items()
    }
    required_values: Set[str] = annotation.extra["required"]

    if verifier := verifier:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Mapping):
                validated: MutableMapping[Any, Any] = {}
                for key, validate in values_validators.items():
                    validated_key: str = key_validator(key)
                    with context.scope(f"[{validated_key}]"):
                        element: Any = value.get(validated_key, MISSING)
                        if element is MISSING and key not in required_values:
                            continue  # skip missing and not required

                        validated[validated_key] = validate(
                            element,
                            context=context,
                        )

                # TODO: FIXME: make sure dict is not mutable?
                # validated = MappingProxyType(validated)
                verifier(validated)
                return validated

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    else:

        def validator(
            value: Any,
            /,
            context: ParameterValidationContext,
        ) -> Any:
            if isinstance(value, Mapping):
                validated: MutableMapping[Any, Any] = {}
                for key, validate in values_validators.items():
                    validated_key: str = key_validator(key)
                    with context.scope(f"[{validated_key}]"):
                        element: Any = value.get(validated_key, MISSING)
                        if element is MISSING and key not in required_values:
                            continue  # skip missing and not required

                        validated[validated_key] = validate(
                            element,
                            context=context,
                        )

                # TODO: FIXME: make sure dict is not mutable?
                # validated = MappingProxyType(validated)
                return validated

            else:
                raise TypeError(f"'{value}' is not matching expected type of '{formatted_type}'")

    return validator


VALIDATORS: Mapping[
    Any,
    Callable[
        [
            AttributeAnnotation,
            ParameterVerification[Any] | None,
            MutableMapping[str, ParameterValidation[Any]],
        ],
        ParameterValidation[Any],
    ],
] = {
    Any: _prepare_validator_of_any,
    NoneType: _prepare_validator_of_none,
    Missing: _prepare_validator_of_missing,
    type: _prepare_validator_of_type,
    bool: _prepare_validator_of_bool,
    int: _prepare_validator_of_int,
    float: _prepare_validator_of_float,
    str: _prepare_validator_of_type,
    tuple: _prepare_validator_of_tuple,
    Literal: _prepare_validator_of_literal,
    Sequence: _prepare_validator_of_sequence,
    Mapping: _prepare_validator_of_mapping,
    Meta: _prepare_validator_of_meta,
    UUID: _prepare_validator_of_uuid,
    date: _prepare_validator_of_date,
    datetime: _prepare_validator_of_datetime,
    time: _prepare_validator_of_time,
    Union: _prepare_validator_of_union,
    UnionType: _prepare_validator_of_union,
}
