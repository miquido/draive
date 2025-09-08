from typing import cast

from haiway import MISSING, Missing
from mistralai import UNSET
from mistralai.types.basemodel import Unset

__all__ = (
    "unwrap_missing_to_none",
    "unwrap_missing_to_unset",
)


def unwrap_missing_to_none[Value](
    value: Value | Missing,
    /,
) -> Value | None:
    if value is MISSING:
        return None
    else:
        return cast(Value, value)


def unwrap_missing_to_unset[Value](
    value: Value | Missing,
    /,
) -> Value | Unset:
    if value is MISSING:
        return UNSET
    else:
        return cast(Value, value)
