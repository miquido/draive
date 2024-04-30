from collections.abc import Callable
from inspect import isfunction
from typing import Protocol, final
from uuid import UUID, uuid4

from draive.agents.errors import AgentException
from draive.agents.state import AgentsChat, AgentsChatMessage, AgentsData
from draive.parameters import ParametrizedData
from draive.scope import ctx
from draive.types import MultimodalContent

__all__ = [
    "agent",
    "Agent",
    "AgentInvocation",
]


class AgentInvocation[Data: ParametrizedData](Protocol):
    async def __call__(
        self,
        chat: AgentsChat,
        data: AgentsData[Data],
    ) -> MultimodalContent: ...


@final
class Agent[Data: ParametrizedData]:
    def __init__(
        self,
        role: str,
        capabilities: str,
        invocation: AgentInvocation[Data],
    ) -> None:
        self.role: str = role
        self.identifier: UUID = uuid4()
        self.capabilities: str = capabilities
        self._invocation: AgentInvocation[Data] = invocation
        self.description: str = (
            f"{self.role}:\n| ID: {self.identifier}\n| Capabilities: {self.capabilities}"
        )

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, Agent):
            return self.identifier == other.identifier
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __str__(self) -> str:
        return self.description

    async def __call__(
        self,
        chat: AgentsChat,
        data: AgentsData[Data],
    ) -> AgentsChatMessage:
        with ctx.nested(
            f"Agent|{self.role}|{self.identifier}",
        ):
            try:
                return AgentsChatMessage(
                    author=self.role,
                    content=await self._invocation(chat, data),
                )

            except Exception as exc:
                raise AgentException(
                    "Agent invocation of %s failed due to an error: %s",
                    self.identifier,
                    exc,
                ) from exc


def agent[Data: ParametrizedData](
    *,
    role: str,
    capabilities: str,
) -> Callable[[AgentInvocation[Data]], Agent[Data]]:
    def wrap(
        invoke: AgentInvocation[Data],
    ) -> Agent[Data]:
        assert isfunction(invoke), "Agent has to be defined from a function"  # nosec: B101
        return Agent[Data](
            role=role,
            capabilities=capabilities,
            invocation=invoke,
        )

    return wrap
