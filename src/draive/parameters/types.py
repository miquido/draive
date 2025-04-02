from collections import deque
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, Protocol

__all__ = (
    "BasicValue",
    "ParameterConversion",
    "ParameterValidation",
    "ParameterValidationContext",
    "ParameterValidationError",
    "ParameterVerification",
)

type BasicValue = (
    Mapping[str, "BasicValue"] | Sequence["BasicValue"] | str | float | int | bool | None
)

type ParameterConversion[Type] = Callable[[Type], BasicValue]
type ParameterVerification[Type] = Callable[[Type], None]


class ParameterValidationContext:
    def __init__(self) -> None:
        self._path: deque[str] = deque()

    def __str__(self) -> str:
        return "".join(self._path)

    def scope(
        self,
        path: str,
        /,
    ) -> "ParameterValidationContextScope":
        return ParameterValidationContextScope(self, component=path)


class ParameterValidationContextScope:
    def __init__(
        self,
        context: ParameterValidationContext,
        /,
        component: str,
    ) -> None:
        self.context: ParameterValidationContext = context
        self.component: str = component

    def __enter__(self) -> ParameterValidationContext:
        self.context._path.append(self.component)
        return self.context

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None and exc_type is not ParameterValidationError:
                raise ParameterValidationError(
                    f"Validation error at {self.context!s}",
                ) from exc_val

        finally:
            self.context._path.pop()


class ParameterValidation[Type](Protocol):
    def __call__(
        self,
        value: Any,
        /,
        *,
        context: ParameterValidationContext,
    ) -> Type: ...


class ParameterValidationError(Exception):
    pass
