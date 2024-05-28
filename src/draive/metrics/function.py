from collections.abc import Sequence
from typing import Any, Self

from draive.parameters import State
from draive.utils import MISSING, Missing

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
        exceptions: Sequence[BaseException]
        exception_messages: Sequence[str]
        if isinstance(self.exception, BaseExceptionGroup):
            exceptions = []
            exception_messages = []
            for exception in (*self.exception.exceptions, other.exception):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                exceptions.append(exception)  # pyright: ignore[reportUnknownArgumentType]
                exception_messages.append(f"{type(exception).__qualname__}:{exception}")  # pyright: ignore[reportUnknownArgumentType]

        else:
            exceptions = (self.exception, other.exception)
            exception_messages = (
                f"{type(self.exception).__qualname__}:{self.exception}",
                f"{type(other.exception).__qualname__}:{other.exception}",
            )

        return self.__class__(
            name="ExceptionGroup",
            exception=BaseExceptionGroup(
                f"Multiple errors: [{','.join(exception_messages)}]",
                exceptions,
            ),
        )
