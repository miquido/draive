from collections.abc import Callable
from typing import cast, overload

from haiway import MISSING, Missing

__all__ = ("unwrap_missing",)


@overload
def unwrap_missing[Value](
    value: Value | Missing,
    /,
    default: Value,
) -> Value: ...


@overload
def unwrap_missing[Value](
    value: Value | Missing,
    /,
    default: Value | None = None,
) -> Value | None: ...


@overload
def unwrap_missing[Value, Result](
    value: Value | Missing,
    /,
    default: Value,
    *,
    transform: Callable[[Value], Result],
) -> Result: ...


@overload
def unwrap_missing[Value, Result](
    value: Value | Missing,
    /,
    default: Value | None = None,
    *,
    transform: Callable[[Value], Result],
) -> Result | None: ...


def unwrap_missing[Value, Result](
    value: Value | Missing,
    /,
    default: Result | Value | None = None,
    *,
    transform: Callable[[Value], Result] | None = None,
) -> Result | Value | None:
    if value is MISSING:
        return default

    elif transform is not None:
        return transform(cast(Value, value))

    else:
        return cast(Result, value)
