from typing import Any, Self

from draive.helpers import MISSING, Missing
from draive.types import State

__all__ = [
    "ArgumentsTrace",
    "ExceptionTrace",
    "ResultTrace",
]


class ArgumentsTrace(State):
    if __debug__:

        @classmethod
        def of(cls, *args: Any, **kwargs: Any) -> Self:
            return cls(
                args=args if args else MISSING,
                kwargs=kwargs if kwargs else MISSING,
            )
    else:

        @classmethod
        def of(cls, *args: Any, **kwargs: Any) -> Self:
            return cls(
                args=MISSING,
                kwargs=MISSING,
            )

    args: tuple[Any, ...] | Missing
    kwargs: dict[str, Any] | Missing


class ResultTrace(State):
    @classmethod
    def of(
        cls,
        value: Any,
        /,
    ) -> Self:
        return cls(result=value)

    result: Any


class ExceptionTrace(State):
    @classmethod
    def of(
        cls,
        exception: BaseException,
        /,
    ) -> Self:
        return cls(
            name=type(exception).__qualname__,
            exception=exception,
        )

    name: str
    exception: BaseException

    def __add__(self, other: Self) -> Self:
        # the code below does not keep proper exception semantics of BaseException/Exception
        # however we are using it only for logging purposes at the moment
        # because of that merging exceptions in groups is simplified under BaseExceptionGroup
        if isinstance(self.exception, BaseExceptionGroup):
            return self.__class__(
                name="ExceptionGroup",
                exception=BaseExceptionGroup(
                    "Multiple errors",
                    (*self.exception.exceptions, other.exception),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                ),
            )
        else:
            return self.__class__(
                name="ExceptionGroup",
                exception=BaseExceptionGroup(
                    "Multiple errors",
                    (self.exception, other.exception),
                ),
            )
