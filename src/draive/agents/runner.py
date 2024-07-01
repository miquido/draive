from asyncio import CancelledError, Task
from collections.abc import AsyncIterator
from typing import Protocol, Self, final, overload, runtime_checkable

from draive.agents.node import Agent, AgentError, AgentMessage, AgentNode
from draive.scope import ctx
from draive.utils import AsyncQueue, freeze

__all__ = [
    "AgentRunner",
    "AgentRunnerOutput",
]


@runtime_checkable
class AgentRunnerOutput(Protocol):
    @overload
    def __call__(
        self,
        error: AgentError,
        /,
    ) -> None: ...

    @overload
    def __call__(
        self,
        messages: AgentMessage,
        /,
        *_messages: AgentMessage,
    ) -> None: ...

    def __call__(
        self,
        messages: AgentMessage | AgentError,
        /,
        *_messages: AgentMessage,
    ) -> None: ...


@final
class AgentRunner:
    @classmethod
    def run(
        cls,
        agent: AgentNode,
        /,
        output: AgentRunnerOutput,
    ) -> Self:
        queue: AsyncQueue[AgentMessage] = AsyncQueue()
        return cls(
            queue=queue,
            task=ctx.spawn_subtask(
                cls._draive,
                agent,
                queue,
                output,
            ),
        )

    def __init__(
        self,
        queue: AsyncQueue[AgentMessage],
        task: Task[None],
    ) -> None:
        self._queue: AsyncQueue[AgentMessage] = queue
        self._task: Task[None] = task

        freeze(self)

    def send(
        self,
        message: AgentMessage,
        /,
    ) -> None:
        self._queue.enqueue(message)

    def cancel(self) -> None:
        self._task.cancel()

    def finish(self) -> None:
        self._queue.cancel()

    async def wait(self) -> None:
        await self._task

    async def finalize(self) -> None:
        self._queue.finish()
        await self._task

    @staticmethod
    async def _draive(
        agent: AgentNode,
        /,
        input: AsyncIterator[AgentMessage],  # noqa: A002
        output: AgentRunnerOutput,
    ) -> None:
        async with ctx.nested(agent.__str__()):
            agent_instance: Agent = agent.initialize()

            try:
                if agent.concurrent:
                    async for message in input:
                        ctx.spawn_subtask(
                            AgentRunner._handle,
                            message,
                            node=agent,
                            agent=agent_instance,
                            output=output,
                        )

                else:
                    async for message in input:
                        await AgentRunner._handle(
                            message,
                            node=agent,
                            agent=agent_instance,
                            output=output,
                        )

            except CancelledError:
                pass  # just finish when Cancelled

    @staticmethod
    async def _handle(
        message: AgentMessage,
        /,
        node: AgentNode,
        agent: Agent,
        output: AgentRunnerOutput,
    ) -> None:
        try:
            match await agent(message=message):
                case None:
                    pass  # nothing to do

                case [*results]:
                    output(*[result.updated(sender=node) for result in results])

                case result:
                    output(result.updated(sender=node))

        except CancelledError:
            pass  # ignore when Cancelled

        except Exception as exc:
            output(
                AgentError(
                    agent=node,
                    message=message,
                    cause=exc,
                )
            )
