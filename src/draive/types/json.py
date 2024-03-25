from typing import Protocol, Self

__all__ = [
    "JSONConvertible",
    "JSONRepresentable",
]


class JSONConvertible(Protocol):
    def as_json(self) -> str:
        ...


class JSONRepresentable(Protocol):
    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        ...

    def as_json(self) -> str:
        ...
