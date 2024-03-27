from typing import Any, Self

from draive.types.parameters import ParametrizedState

__all__ = [
    "State",
]


class State(ParametrizedState):
    # TODO: find a way to generate signature similar to dataclass __init__
    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__class__.validated(**{**vars(self), **kwargs})
