from abc import ABC, abstractmethod

from draive.agents.state import AgentScratchpad, AgentState
from draive.parameters import ParametrizedData
from draive.scope import ctx
from draive.types import MultimodalContent

__all__ = [
    "BaseAgent",
]


class BaseAgent[State: ParametrizedData](ABC):
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
    ) -> None:
        self.agent_id: str = agent_id
        self.name: str = name
        self.description: str = description

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, BaseAgent):
            return self.agent_id == other.agent_id
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.agent_id)

    @abstractmethod
    async def __call__(
        self,
        state: AgentState[State],
    ) -> MultimodalContent | None: ...

    async def run(
        self,
        initial_state: State,
        scratchpad: MultimodalContent | None = None,
    ) -> State:
        state: AgentState[State] = AgentState(initial=initial_state)

        with ctx.updated(AgentScratchpad.prepare(scratchpad)):
            await self(state)

        return await state.current
