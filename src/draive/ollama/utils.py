from typing import cast

from haiway import MISSING, Missing

__all__ = ("unwrap_missing",)


def unwrap_missing[Value](
    value: Value | Missing,
    /,
) -> Value | None:
    if value is MISSING:
        return None
    else:
        return cast(Value, value)
