from asyncio import Task
from typing import Self, final
from uuid import UUID

from draive.agents.types import AgentInput, AgentMessage, AgentWorkflowCurrent, WorkflowAgentBase
from draive.parameters import ParametrizedData
from draive.scope import ctx
from draive.utils import AsyncStream, freeze

__all__ = [
    "AgentRunner",
]


@final
class AgentRunner:
    @classmethod
    def spawn[
        WorkflowState: ParametrizedData,
        WorkflowResult,
    ](
        cls,
        agent: WorkflowAgentBase[WorkflowState, WorkflowResult],
        /,
        workflow: AgentWorkflowCurrent[WorkflowState, WorkflowResult],
    ) -> Self:
        agent_input: AgentInput = AsyncStream()

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
        input: AgentInput,  # noqa: A002
        task: Task[None],
    ) -> None:
        self.identifier: UUID = identifier
        self._input: AgentInput = input
        self._task: Task[None] = task

        freeze(self)

    def send(
        self,
        message: AgentMessage,
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
