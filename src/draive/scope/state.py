from collections.abc import Iterable
from typing import Self, TypeVar, cast, final

from draive.scope.errors import MissingScopeState
from draive.types.parameters import ParametrizedState

__all__ = [
    "ScopeState",
]


_State_T = TypeVar(
    "_State_T",
    bound=ParametrizedState,
)


@final
class ScopeState:
    def __init__(
        self,
        *state: ParametrizedState,
    ) -> None:
        self._state: dict[type[ParametrizedState], ParametrizedState] = {
            type(element): element for element in state
        }

    def state(
        self,
        state: type[_State_T],
        /,
    ) -> _State_T:
        if state in self._state:
            return cast(_State_T, self._state[state])
        else:
            try:
                default: _State_T = state()
                self._state[state] = default
                return default
            except (TypeError, AttributeError) as exc:
                raise MissingScopeState(
                    f"{state.__qualname__} is not defined in the current scope"
                    " and failed to provide a default value"
                ) from exc

    def updated(
        self,
        state: Iterable[ParametrizedState] | None,
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
