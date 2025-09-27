from collections.abc import Callable
from typing import cast, overload

from anthropic import Omit, omit
from haiway import MISSING, Missing

__all__ = ("unwrap_missing",)


@overload
def unwrap_missing[Value](
    value: Value | Missing,
    /,
) -> Value | Omit: ...


@overload
def unwrap_missing[Value, Converted](
    value: Value | Missing,
    /,
    convert: Callable[[Value], Converted],
) -> Converted | Omit: ...


def unwrap_missing[Value, Converted](
    value: Value | Missing,
    /,
    convert: Callable[[Value], Converted] | None = None,
) -> Converted | Value | Omit:
    if value is MISSING:
        return omit

    elif convert is not None:
        return convert(cast(Value, value))

    else:
        return cast(Value, value)
