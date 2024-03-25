from typing import Any, Protocol, Self, runtime_checkable

__all__ = [
    "DictionaryConvertible",
    "DictionaryRepresentable",
]


class _DictionaryConvertible(Protocol):
    def as_dict(self) -> dict[str, Any]:
        ...


DictionaryConvertible = _DictionaryConvertible | dict[str, Any]


@runtime_checkable
class DictionaryRepresentable(Protocol):
    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        ...

    def as_dict(self) -> dict[str, Any]:
        ...
