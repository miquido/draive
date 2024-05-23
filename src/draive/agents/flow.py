from asyncio import gather
from typing import final
from uuid import uuid4

from draive.agents.abc import BaseAgent
from draive.agents.state import AgentScratchpad, AgentState
from draive.parameters import ParametrizedData
from draive.scope import ctx
from draive.types import MultimodalContent
from draive.utils import freeze

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
        assert agents, "Can't make emptty agent flow"  # nosec: B101
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
    ) -> MultimodalContent:
        current_scratchpad: AgentScratchpad = AgentScratchpad.current()
        scratchpad_notes: list[MultimodalContent] = []
        for agent in self.agents:
            with ctx.updated(current_scratchpad):
                match agent:
                    case [*agents]:
                        merged_note: MultimodalContent = MultimodalContent.of(
                            *[
                                scratchpad_note
                                for scratchpad_note in await gather(
                                    *[agent(state) for agent in agents],
                                )
                                if scratchpad_note is not None
                            ]
                        )
                        if merged_note:
                            scratchpad_notes.append(merged_note)
                            current_scratchpad = current_scratchpad.extended(merged_note)

                    case agent:
                        scratchpad_note: MultimodalContent | None = await agent(state)
                        if scratchpad_note:
                            current_scratchpad = current_scratchpad.extended(scratchpad_note)
                            scratchpad_notes.append(scratchpad_note)

        return MultimodalContent.of(*scratchpad_notes)
