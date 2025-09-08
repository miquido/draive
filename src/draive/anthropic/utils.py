from collections.abc import Callable
from typing import cast, overload

from anthropic import NOT_GIVEN, NotGiven
from haiway import MISSING, Missing

__all__ = ("unwrap_missing",)


@overload
def unwrap_missing[Value](
    value: Value | Missing,
    /,
) -> Value | NotGiven: ...


@overload
def unwrap_missing[Value, Converted](
    value: Value | Missing,
    /,
    convert: Callable[[Value], Converted],
) -> Converted | NotGiven: ...


def unwrap_missing[Value, Converted](
    value: Value | Missing,
    /,
    convert: Callable[[Value], Converted] | None = None,
) -> Converted | Value | NotGiven:
    if value is MISSING:
        return NOT_GIVEN

    elif convert is not None:
        return convert(cast(Value, value))

    else:
        return cast(Value, value)
