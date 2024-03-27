from dataclasses import asdict
from typing import Any, Self

from draive.types.parameters import ParametrizedState

__all__ = [
    "State",
]


class State(ParametrizedState):
    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> Self:
        try:
            return cls.validated(**values)
        except Exception as exc:
            raise ValueError(f"Failed to decode {cls.__name__} from dict:\n{values}") from exc

    def as_dict(self) -> dict[str, Any]:
        return self.__class__.aliased_parameters(asdict(self))

    # TODO: find a way to generate signature similar to dataclass __init__
    def updated(
        self,
        **kwargs: Any,
    ) -> Self:
        return self.__class__.validated(**{**vars(self), **kwargs})
