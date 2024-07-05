__all__ = [
    "noop",
]


from typing import Any


async def noop(
    *args: Any,
    **kwargs: Any,
) -> None:
    pass
