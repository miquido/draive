from asyncio import Task
from typing import Self, final
from uuid import UUID

from draive.agents.types import AgentID, AgentInput, AgentMessage, AgentWorkflowBase
from draive.scope import ctx
from draive.utils import AsyncStream, freeze

__all__ = [
    "WorkflowAgentRunner",
]


@final
class WorkflowAgentRunner[Workflow]:
    @classmethod
    def spawn(
        cls,
        agent: AgentID[Workflow],
        /,
        workflow: AgentWorkflowBase[Workflow],
    ) -> Self:
        agent_input: AgentInput[Workflow] = AsyncStream()

        return cls(
            identifier=agent.identifier,
            input=agent_input,
            task=ctx.spawn_subtask(
                agent._draive,  # pyright: ignore[reportPrivateUsage]
                agent_input,
                workflow,
            ),
        )

    def __init__(
        self,
        identifier: UUID,
        input: AgentInput[Workflow],  # noqa: A002
        task: Task[None],
    ) -> None:
        self.identifier: UUID = identifier
        self._input: AgentInput[Workflow] = input
        self._task: Task[None] = task

        freeze(self)

    def send(
        self,
        message: AgentMessage[Workflow],
        /,
    ) -> None:
        self._input.send(message)

    def finish(self) -> None:
        if not self._input.finished:
            self._input.finish()

    async def finalize(self) -> None:
        self.finish()

        await self._input.wait()
        await self._task
