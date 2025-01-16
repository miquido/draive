from collections.abc import Iterable
from typing import Any

from haiway import State

from draive.parameters import DataModel

__all__ = [
    "StageState",
]


class StageState:
    def __init__(
        self,
        state: Iterable[DataModel | State] | None,
    ) -> None:
        self._state: dict[type[DataModel | State], Any]
        object.__setattr__(
            self,
            "_state",
            {type(element): element for element in state} if state else {},
        )

    async def read[StateType: DataModel | State](
        self,
        state: type[StateType],
        /,
        default: StateType | None = None,
    ) -> StateType | None:
        if state in self._state:
            return self._state[state]

        else:
            return default

    async def write(
        self,
        state: DataModel | State,
        /,
    ) -> None:
        self._state[type(state)] = state

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError("WorkflowState is frozen and can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError("WorkflowState is frozen and can't be modified")
