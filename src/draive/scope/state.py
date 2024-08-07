from collections.abc import Iterable
from typing import Self, cast, final

from draive.parameters import ParametrizedData
from draive.scope.errors import MissingScopeState
from draive.utils import MISSING, Missing, not_missing

__all__ = [
    "ScopeState",
]


@final
class ScopeState:
    def __init__(
        self,
        *state: ParametrizedData,
    ) -> None:
        self._state: dict[type[ParametrizedData], ParametrizedData] = {
            type(element): element for element in state
        }

    def state[State_T: ParametrizedData](
        self,
        state: type[State_T],
        /,
        default: State_T | Missing = MISSING,
    ) -> State_T:
        if state in self._state:
            return cast(State_T, self._state[state])

        elif not_missing(default):
            return default

        else:
            try:
                initialized: State_T = state()
                self._state[state] = initialized
                return initialized

            except (TypeError, AttributeError) as exc:
                raise MissingScopeState(
                    f"{state.__qualname__} is not defined in the current scope"
                    " and failed to provide a default value"
                ) from exc

    def updated(
        self,
        state: Iterable[ParametrizedData] | None,
    ) -> Self:
        if state:
            return self.__class__(
                *[
                    *self._state.values(),
                    *state,
                ]
            )
        else:
            return self
