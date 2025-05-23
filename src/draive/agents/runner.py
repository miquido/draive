from asyncio import (
    CancelledError,
    Task,
)
from collections.abc import AsyncIterator
from typing import Protocol, Self, final, overload, runtime_checkable

from haiway import AsyncQueue, ctx

from draive.agents.idle import IdleMonitor
from draive.agents.node import Agent, AgentError, AgentMessage, AgentNode

__all__ = (
    "AgentRunner",
    "AgentRunnerOutput",
)


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
        status: IdleMonitor,
    ) -> Self:
        queue: AsyncQueue[AgentMessage] = AsyncQueue()
        return cls(
            queue=queue,
            task=ctx.spawn(
                cls._draive,
                agent,
                queue,
                output,
                status,
            ),
            status=status,
        )

    def __init__(
        self,
        queue: AsyncQueue[AgentMessage],
        task: Task[None],
        status: IdleMonitor,
    ) -> None:
        self._queue: AsyncQueue[AgentMessage] = queue
        self._task: Task[None] = task
        self._status: IdleMonitor = status

    def send(
        self,
        message: AgentMessage,
        /,
    ) -> None:
        # not idle when received a message
        self._status.enter_task()
        self._queue.enqueue(message)

    @property
    def idle(self) -> bool:
        return self._status.idle

    async def wait_idle(self) -> None:
        await self._status.wait_idle()

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
        status: IdleMonitor,
    ) -> None:
        async with ctx.scope(agent.__str__()):  # pyright: ignore[reportDeprecated]
            agent_instance: Agent = agent.initialize()

            try:
                if agent.concurrent:
                    async for message in input:
                        task: Task[None] = ctx.spawn(
                            AgentRunner._handle,
                            message,
                            node=agent,
                            agent=agent_instance,
                            output=output,
                        )
                        task.add_done_callback(status.exit_task)

                else:
                    async for message in input:
                        try:
                            await AgentRunner._handle(
                                message,
                                node=agent,
                                agent=agent_instance,
                                output=output,
                            )

                        finally:
                            status.exit_task()

            except CancelledError:
                pass  # just end on cancel

            finally:  # cleanup status when exiting
                status.exit_all()

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
                    output(*[result._with_sender(node) for result in results])  # pyright: ignore[reportPrivateUsage]

                case result:
                    output(result._with_sender(node))  # pyright: ignore[reportPrivateUsage]

        except CancelledError:
            pass  # ignore when Cancelled

        except Exception as exc:
            output(
                AgentError(
                    "Agent failed while processing the message",
                    agent=node,
                    message=message,
                    cause=exc,
                )
            )
