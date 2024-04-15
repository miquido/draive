from typing import Any, Self

from draive.metrics.metric import Metric
from draive.types import MISSING, MissingValue

__all__ = [
    "ArgumentsTrace",
    "ExceptionTrace",
    "ResultTrace",
]


class ArgumentsTrace(Metric):
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

    args: tuple[Any, ...] | MissingValue
    kwargs: dict[str, Any] | MissingValue


class ResultTrace(Metric):
    @classmethod
    def of(
        cls,
        value: Any,
        /,
    ) -> Self:
        return cls(result=value)

    result: Any


class ExceptionTrace(Metric):
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
                    (*self.exception.exceptions, other.exception),  # pyright: ignore[reportUnknownMemberType]
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
