from asyncio import gather
from typing import final
from uuid import uuid4

from draive.agents.abc import BaseAgent
from draive.agents.state import AgentState
from draive.helpers import freeze
from draive.parameters import ParametrizedData

__all__ = [
    "AgentFlow",
]


@final
class AgentFlow[State: ParametrizedData](BaseAgent[State]):
    def __init__(
        self,
        *agents: tuple[BaseAgent[State], ...] | BaseAgent[State],
        name: str,
        description: str,
    ) -> None:
        super().__init__(
            agent_id=uuid4().hex,
            name=name,
            description=description,
        )
        self.agents: tuple[tuple[BaseAgent[State], ...] | BaseAgent[State], ...] = agents

        freeze(self)

    async def __call__(
        self,
        state: AgentState[State],
    ) -> None:
        for agent in self.agents:
            match agent:
                case [*agents]:
                    await gather(
                        *[agent(state) for agent in agents],
                    )

                # case [agent]:
                #     await agent(state)

                case agent:
                    await agent(state)
