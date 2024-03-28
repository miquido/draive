from collections.abc import Iterable
from contextvars import Token
from typing import Self, TypeVar, cast, final

from draive.scope.errors import MissingScopeState
from draive.types.parameters import ParametrizedState

__all__ = [
    "_ScopeState_T",
    "StateScope",
]


_ScopeState_T = TypeVar(
    "_ScopeState_T",
    bound=ParametrizedState,
)


@final
class StateScope:
    def __init__(
        self,
        *state: ParametrizedState,
    ) -> None:
        self._state: dict[type[ParametrizedState], ParametrizedState] = {
            type(element): element for element in state
        }
        self._token: Token[StateScope] | None = None

    def state(
        self,
        _type: type[_ScopeState_T],
        /,
    ) -> _ScopeState_T:
        if _type in self._state:
            return cast(_ScopeState_T, self._state[_type])
        else:
            try:
                default: _ScopeState_T = _type()
                self._state[_type] = default
                return default
            except (TypeError, AttributeError) as exc:
                raise MissingScopeState(
                    f"{_type} is not defined in current scope and failed to provide a default value"
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
