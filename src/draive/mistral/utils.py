from collections.abc import Sequence
from typing import cast

from haiway import Missing
from mistralai.client import UNSET
from mistralai.client.types.basemodel import Unset

__all__ = (
    "unwrap_missing_list_to_unset",
    "unwrap_missing_to_unset",
)


def unwrap_missing_to_unset[Value](
    value: Value | Missing,
    /,
) -> Value | Unset:
    if value is Missing:
        return UNSET
    else:
        return cast(Value, value)


def unwrap_missing_list_to_unset[Value](
    value: Sequence[Value] | Missing,
    /,
) -> list[Value] | Unset:
    if value is Missing:
        return UNSET
    else:
        return list(value)
