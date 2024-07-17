from typing import Any, Self

from draive.utils import freeze

__all__ = [
    "ParameterValidationContext",
    "ParameterValidationError",
]


class ParameterValidationContext:
    def __init__(
        self,
        path: tuple[str, ...],
    ) -> None:
        self._path: tuple[str, ...] = path

        freeze(self)

    def __str__(self) -> str:
        return "".join(self._path)

    def __repr__(self) -> str:
        return "".join(self._path)

    def appending_path(
        self,
        path: str,
        /,
    ) -> Self:
        return self.__class__(
            path=(*self._path, path),
        )


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
            getattr(expected, "__name__", str(expected)),
            getattr(type(received), "__name__", str(received)),  # pyright: ignore[reportUnknownArgumentType]
        )

    @classmethod
    def invalid_union_type(
        cls,
        expected: tuple[Any, ...],
        received: Any,
        context: ParameterValidationContext,
    ) -> Self:
        return cls(
            "Validation error: Invalid parameter at: %s - expected %s while received '%s'",
            context,
            " | ".join(getattr(t, "__name__", str(t)) for t in expected),
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
