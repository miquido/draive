from typing import Generic, TypeVar

from draive.types.state import State

__all__ = [
    "Embedded",
]


_Embedded = TypeVar(
    "_Embedded",
    bound=object,
)


class Embedded(State, Generic[_Embedded]):
    value: _Embedded
    vector: list[float]
