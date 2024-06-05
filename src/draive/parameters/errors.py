from typing import Any, Self

__all__ = [
    "ParameterValidationContext",
    "ParameterValidationError",
]

type ParameterValidationContext = tuple[str, ...]


class ParameterValidationError(Exception):
    @classmethod
    def missing(
        cls,
        context: ParameterValidationContext,
    ) -> Self:
        return cls("Validation error: Missing required parameter at: %s", context)

    @classmethod
    def invalid_type(
        cls,
        expected: Any,
        received: Any,
        context: ParameterValidationContext,
    ) -> Self:
        return cls(
            "Validation error: Invalid parameter at: %s - expected '%s' while received '%s'",
            context,
            expected,
            received,
        )

    @classmethod
    def invalid_key(
        cls,
        received: Any,
        context: ParameterValidationContext,
    ) -> Self:
        return cls(
            "Validation error: Invalid parameter key at: %s - received '%s'",
            context,
            received,
        )

    @classmethod
    def invalid_value(
        cls,
        expected: Any,
        received: Any,
        context: ParameterValidationContext,
    ) -> Self:
        return cls(
            "Validation error: Invalid parameter value at: %s - expected '%s' while received '%s'",
            context,
            expected,
            received,
        )

    @classmethod
    def invalid(
        cls,
        exception: Exception,
        context: ParameterValidationContext,
    ) -> Self:
        return cls(
            "Validation error: Invalid parameter at: %s - %s",
            context,
            exception,
        )
