from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Event,
    Future,
    TimerHandle,
    get_running_loop,
)
from typing import Any, Final, Self, final
from uuid import UUID, uuid4

from draive.agents.runner import WorkflowAgentRunner
from draive.agents.types import AgentID, AgentMessage, AgentWorkflowBase, WorkflowState
from draive.parameters import ParametrizedData
from draive.scope import ctx
from draive.utils import AsyncStream, freeze

__all__ = [
    "AgentWorkflow",
]

WORKFLOW_ENTRY: Final[AgentID[Any]] = AgentID(Any, identifier=UUID(int=0))


@final
class AgentWorkflow[Workflow](AgentWorkflowBase[Workflow]):
    @classmethod
    async def run(
        cls,
        messages: AgentMessage[Workflow],
        /,
        *_messages: AgentMessage[Workflow],
        state: tuple[ParametrizedData, ...] | ParametrizedData,
        timeout: float = 600,  # default timeout is 10 minutes
    ) -> Workflow:
        workflow: Self = cls(state=state)

        async with ctx.nested(f"Workflow|{workflow.identifier}"):
            workflow.send(
                messages.updated(sender=WORKFLOW_ENTRY),
                *(message.updated(sender=WORKFLOW_ENTRY) for message in _messages),
            )

            return await workflow.execute(timeout=timeout)

    def __init__(
        self,
        state: tuple[ParametrizedData, ...] | ParametrizedData,
    ) -> None:
        self._workflow_state: WorkflowState = WorkflowState(state)
        self.identifier: UUID = uuid4()
        self._runners: dict[UUID, WorkflowAgentRunner] = {}
        self._messages: AsyncStream[AgentMessage[Workflow]] = AsyncStream()
        self._message_trace: dict[UUID, Event] = {}
        self._result: Future[Workflow] = Future()

        freeze(self)

    def __del__(self) -> None:
        self.finish_with(CancelledError())

    async def _deliver(
        self,
        message: AgentMessage[Workflow],
        /,
    ) -> None:
        assert message.recipient != WORKFLOW_ENTRY  # nosec: B101

        ctx.log_info(
            "Delivering message [%s] from %s to %s",
            message.identifier,
            message.sender,
            message.recipient,
        )
        if __debug__:
            ctx.log_debug(
                "\nMessage [%s] content: %s",
                message.identifier,
                message.content,
            )

        runner: WorkflowAgentRunner
        if running := self._runners.get(message.recipient.identifier):
            runner = running

        else:
            runner = WorkflowAgentRunner.spawn(
                message.recipient,
                workflow=self,
            )
            self._runners[message.recipient.identifier] = runner

        runner.send(message)

    def send(
        self,
        messages: AgentMessage[Workflow],
        /,
        *_messages: AgentMessage[Workflow],
    ) -> None:
        if self._result.done():
            return ctx.log_debug("Ignoring messages - workflow finished")

        self._messages.send(messages, *_messages)

    def finish_with(
        self,
        result: Workflow | BaseException,
        /,
    ) -> None:
        if self._result.done():
            return  # ignore - already done

        for runner in self._runners.values():
            runner.finish()

        self._messages.finish()

        match result:
            case BaseException() as exception:
                self._result.set_exception(exception)

            case result:
                self._result.set_result(result)

    async def execute(
        self,
        timeout: float = 600,  # default timeout is 10 minutes
    ) -> Workflow:
        if self._result.done():
            raise RuntimeError("AgentWorkflow can be executed only once!")

        loop: AbstractEventLoop = get_running_loop()

        def on_timeout(
            future: Future[Workflow],
        ) -> None:
            if future.done():
                return  # ignore if already finished

            # result future on its completion will ensure that task will complete
            future.set_exception(TimeoutError())

        timeout_handle: TimerHandle = loop.call_later(
            timeout,
            on_timeout,
            self._result,
        )

        def on_result(
            future: Future[Workflow],
        ) -> None:
            timeout_handle.cancel()  # at this stage we no longer need timeout to trigger
            self._messages.finish()

        self._result.add_done_callback(on_result)

        # TODO: detect idle state - no agent working
        async for message in self._messages:
            await self._deliver(message)

        return await self._result
