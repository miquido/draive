from typing import Any, Protocol, Self

__all__ = [
    "DictionaryConvertible",
]


class _DictionaryConvertible(Protocol):
    def as_dict(self) -> dict[str, Any]:
        ...


DictionaryConvertible = _DictionaryConvertible | dict[str, Any]


class DictionaryRepresentable(Protocol):
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        ...

    def as_dict(self) -> dict[str, Any]:
        ...
