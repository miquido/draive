from typing import Protocol

from draive.agents.state import AgentState
from draive.parameters import ParametrizedData
from draive.types import MultimodalContent

__all__ = [
    "AgentInvocation",
]


class AgentInvocation[State: ParametrizedData](Protocol):
    async def __call__(
        self,
        state: AgentState[State],
    ) -> MultimodalContent | None: ...
