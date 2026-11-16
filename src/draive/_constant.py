from collections.abc import Callable, Mapping
from typing import Any, cast, overload

__all__ = ("class_constant",)


class _ClassConstant[Value]:
    __slots__ = ("_args", "_factory", "_kwargs", "_value")

    def __init__(
        self,
        factory: Callable[..., Value],
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        self._factory: Callable[..., Value] = factory
        self._args: tuple[Any, ...] = args
        self._kwargs: Mapping[str, Any] = kwargs
        self._value: Value

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[Any],
        /,
    ) -> Value: ...

    @overload
    def __get__(
        self,
        instance: object,
        owner: type[Any] | None = None,
        /,
    ) -> Value: ...

    def __get__(
        self,
        instance: object | None,
        owner: type[Any] | None = None,
        /,
    ) -> Value:
        try:
            return self._value

        except AttributeError:
            self._value = self._factory(*self._args, **self._kwargs)
            return self._value


def class_constant[Value, **Args](
    factory: Callable[Args, Value],
    /,
    *args: Args.args,
    **kwargs: Args.kwargs,
) -> Value:
    """Declare a class level constant created on first access.

    A class can't be changed once it is created and its body can't build an instance of the
    class itself. The factory is called with the given arguments once, on the first access, when
    the class is complete. It has to be defined before the class and may refer to the class in its
    body and, with deferred annotations, in its return annotation.

    The result is typed as the value the factory produces, so the attribute is declared with the
    type of that value, i.e. ``empty: ClassVar[Content] = class_constant(_content, parts=())``.

    Parameters
    ----------
    factory : Callable[Args, Value]
        Creates the value, it is called at most once.
    *args : Args.args
        Positional arguments of the factory.
    **kwargs : Args.kwargs
        Keyword arguments of the factory.

    Returns
    -------
    Value
        A descriptor resolving to the value, typed as the value itself.
    """
    return cast(Value, _ClassConstant(factory, args, kwargs))
