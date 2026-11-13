from collections.abc import Callable
from typing import Any, cast, overload

__all__ = ("class_constant",)


class _ClassConstant[Value]:
    """Class level constant created on first access.

    A class can't be changed once it is created and its body can't build an instance of the
    class itself. The factory is called once, on the first access, when the class is complete.

    Parameters
    ----------
    factory : Callable[[], Value]
        Creates the value, it is called at most once.
    """

    __slots__ = ("_factory", "_value")

    def __init__(
        self,
        factory: Callable[[], Value],
        /,
    ) -> None:
        self._factory: Callable[[], Value] = factory
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
            self._value = self._factory()
            return self._value


def class_constant[Value](
    factory: Callable[[], Value],
    /,
) -> Value:
    """Declare a class level constant created on first access.

    A class can't be changed once it is created and its body can't build an instance of the
    class itself. The factory is called once, on the first access, when the class is complete.
    The result is typed as the value it produces, so the attribute is declared with the type of
    that value, i.e. ``empty: ClassVar[Content] = class_constant(lambda: Content(...))``.

    Parameters
    ----------
    factory : Callable[[], Value]
        Creates the value, it is called at most once.

    Returns
    -------
    Value
        A descriptor resolving to the value, typed as the value itself.
    """
    return cast(Value, _ClassConstant(factory))
