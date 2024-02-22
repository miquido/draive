from collections.abc import Iterable
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Self, TypeVar, cast, final

from draive.scope.errors import MissingScopeState
from draive.types.state import State

__all__ = [
    "ScopeState",
    "_ScopeState_T",
    "ScopeStates",
]


class ScopeState(State):
    @classmethod
    def default(cls) -> Self:
        try:
            return cls()
        except AttributeError as exc:
            raise MissingScopeState(
                f"{cls} is not defined in current scope and does not provide a default value"
            ) from exc


_ScopeState_T = TypeVar(
    "_ScopeState_T",
    bound=ScopeState,
)


@final
class ScopeStates:
    def __init__(
        self,
        *state: ScopeState,
    ) -> None:
        self._state: dict[type[ScopeState], ScopeState] = {
            type(element): element for element in state
        }

    def __copy__(self) -> Self:
        return self.__class__(*self._state.values())

    def state(
        self,
        _type: type[_ScopeState_T],
        /,
    ) -> _ScopeState_T:
        if _type in self._state:
            return cast(_ScopeState_T, self._state[_type])
        else:
            self._state[_type] = _type.default()
            return cast(_ScopeState_T, self._state[_type])

    def updated(
        self,
        state: Iterable[ScopeState],
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

    def __enter__(self) -> None:
        assert not hasattr(self, "_token"), "Reentrance is not allowed"  # nosec: B101
        self._token: Token[ScopeStates] = _ScopeState_Var.set(self)

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ScopeState_Var.reset(self._token)
        del self._token


_ScopeState_Var = ContextVar[ScopeStates]("_ScopeState_Var")
