from typing import cast

from haiway import Missing

__all__ = ("unwrap_missing",)


def unwrap_missing[Value](
    value: Value | Missing,
    /,
) -> Value | None:
    if value is Missing:
        return None
    else:
        return cast(Value, value)
