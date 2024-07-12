from typing import Any

__all__ = [
    "async_noop",
    "noop",
]


def noop(
    *args: Any,
    **kwargs: Any,
) -> None:
    pass


async def async_noop(
    *args: Any,
    **kwargs: Any,
) -> None:
    pass
