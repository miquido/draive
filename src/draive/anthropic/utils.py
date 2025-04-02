from typing import cast

from anthropic import NOT_GIVEN, NotGiven
from haiway import MISSING, Missing

__all__ = ("unwrap_missing",)


def unwrap_missing[Value, Default](
    value: Value | Missing,
    /,
) -> Value | NotGiven:
    if value is MISSING:
        return NOT_GIVEN
    else:
        return cast(Value, value)
