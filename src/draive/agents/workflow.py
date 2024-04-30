from collections.abc import Callable
from typing import final
from uuid import UUID

from draive.agents.agent import Agent
from draive.agents.coordinator import AgentDisposition, AgentsCoordinator, basic_agent_coordinator
from draive.agents.state import AgentsChat, AgentsChatMessage, AgentsData
from draive.helpers import freeze
from draive.parameters import ParametrizedData
from draive.types import MultimodalContent

__all__ = [
    "AgentsWorkflow",
]

INITIAL_AGENT_IDENTIFIER: UUID = UUID(int=0)


@final
class AgentsWorkflow[Data: ParametrizedData]:
    def __init__(
        self,
        *entrypoints: Agent[Data],
        state_initializer: Callable[[], Data],
        coordinator: AgentsCoordinator[Data] | None = None,
    ) -> None:
        self._destinations: dict[UUID, frozenset[Agent[Data]]] = {
            INITIAL_AGENT_IDENTIFIER: frozenset(entrypoints),
            **{entrypoint.identifier: frozenset() for entrypoint in entrypoints},
        }
        self.state_initializer: Callable[[], Data] = state_initializer
        self._coordinator: AgentsCoordinator[Data] = coordinator or basic_agent_coordinator

        freeze(self)

    def connect(
        self,
        source: Agent[Data],
        destination: Agent[Data],
    ) -> None:
        assert source.identifier in self._destinations, (  # nosec: B101
            "Can't connect an agent outside of currently defined agents"
        )
        if destination.identifier not in self._destinations:
            self._destinations[destination.identifier] = frozenset()
        self._destinations[source.identifier] = frozenset(
            [
                *self._destinations[source.identifier],
                destination,
            ],
        )

    async def __call__(
        self,
        input: MultimodalContent,  # noqa: A002
        /,
        state: Data | None = None,
    ) -> Data:
        assert any(  # nosec: B101
            not destinations for destinations in self._destinations.values()
        ), "There is no final agent defined, workflow will never end"
        chat: AgentsChat = AgentsChat(goal=input)
        data: AgentsData[Data] = AgentsData(initial=state or self.state_initializer())
        current_agent: UUID = INITIAL_AGENT_IDENTIFIER
        while destinations := self._destinations.get(current_agent):
            disposition: AgentDisposition[Data] = await self._coordinator(
                chat=chat,
                data=data,
                agents=destinations,
            )
            current_agent = disposition.recipient.identifier
            chat = chat.appending(
                message=AgentsChatMessage(
                    author="Coordinator",
                    content=disposition.message,
                )
            )
            response: AgentsChatMessage = await disposition.recipient(
                chat=chat,
                data=data,
            )
            chat = chat.appending(message=response)

        return await data.current_data
