from collections.abc import Callable, Coroutine
from typing import Any

__all__ = [
    "always",
    "async_always",
]


def always[Value](
    value: Value,
    /,
) -> Callable[..., Value]:
    def always_value(
        *args: Any,
        **kwargs: Any,
    ) -> Value:
        return value

    return always_value


def async_always[Value](
    value: Value,
    /,
) -> Callable[..., Coroutine[None, None, Value]]:
    async def always_value(
        *args: Any,
        **kwargs: Any,
    ) -> Value:
        return value

    return always_value
