from typing import cast

from haiway import MISSING, Missing
from openai import Omit, omit

__all__ = ("unwrap_missing",)


def unwrap_missing[Value](
    value: Value | Missing,
    /,
) -> Value | Omit:
    if value is MISSING:
        return omit

    else:
        return cast(Value, value)
