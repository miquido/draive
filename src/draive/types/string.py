from typing import Protocol

__all__ = [
    "StringConvertible",
]


class _StringConvertible(Protocol):
    def __str__(self) -> str:
        ...


StringConvertible = _StringConvertible | str
