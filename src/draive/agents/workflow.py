from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Event,
    Future,
    Task,
    TimerHandle,
    get_running_loop,
)
from inspect import isfunction
from typing import Any, Final, Protocol, Self, final, runtime_checkable
from uuid import UUID, uuid4

from draive.agents.agent import Agent, AgentID, AgentInput, AgentMessage
from draive.agents.errors import AgentException
from draive.agents.workflow import AgentWorkflow
from draive.scope import ctx
from draive.types import frozenlist
from draive.types.memory import Memory
from draive.utils import AsyncStream, freeze

__all__ = [
    "AgentWorkflow",
]

WORKFLOW_ENTRY: Final[AgentID] = AgentID(
    UUID(int=0),
    name="WORKFLOW_ENTRY",
    capabilities=None,
)


@final
class WorkflowAgentRunner[WorkflowResult]:
    @classmethod
    def spawn(
        cls,
        agent: AgentID,
        /,
        workflow: "AgentWorkflow[WorkflowResult]",
    ) -> Self: ...

    def __init__(
        self,
        agent: AgentID,
        task: Task[None],
    ) -> None:
        self.agent: AgentID = agent
        self._input: AgentInput = AsyncStream()
        self._task: Task[None] = ctx.spawn_subtask(
            agent._draive,  # pyright: ignore[reportPrivateUsage]
            agent_input,
            workflow,
        )

        freeze(self)

    def send(
        self,
        message: AgentMessage,
        /,
    ) -> None:
        self._input.send(message)

    def cancel(self) -> None:
        self._task.cancel()
        if not self._input.finished:
            self._input.finish(exception=CancelledError())

    async def finalize(self) -> None:
        if not self._input.finished:
            self._input.finish()

        await self._input.wait()
        await self._task

    @classmethod
    async def _draive(  # yes, it is the name of the library
        cls,
        input: AgentInput,  # noqa: A002
        agent: Agent[Any, Any],
        workflow: AgentWorkflow[WorkflowResult],
    ) -> None:
        with ctx.nested(agent.__str__()):
            memory: Memory[Any, Any] = agent.prepare_memory()
            try:
                if agent._concurrent:  # pyright: ignore[reportPrivateUsage]
                    # process all messages concurrently by spawning a task for each
                    async for message in input:
                        ctx.spawn_subtask(
                            agent.process_message,
                            message,
                            memory,
                        )

                else:
                    # process only one message at the time by waiting for results
                    async for message in input:
                        await agent.process_message(
                            message,
                            memory=memory,
                        )

            except Exception as exc:
                workflow.fail_with(AgentException(cause=exc))

    async def _handle_message(
        self,
        message: AgentMessage,
        /,
        workflow: AgentWorkflow,
    ) -> None:
        try:
            self._handle_output(
                await self(
                    current=agent_current,
                    message=message,
                ),
                workflow=workflow,
            )

        except Exception as exc:  # when any agent fails the whole workflow fails
            workflow.finish_with(AgentException(cause=exc))

    def _handle_output(
        self,
        output: AgentOutput[WorkflowResult],
        /,
        workflow: AgentWorkflowCurrent[WorkflowState, WorkflowResult],
    ) -> None:
        match output:
            case None:
                pass  # no action

            case AgentMessageDraftGroup() as messages_group:
                workflow.send(
                    *(
                        AgentMessage(
                            sender=self,
                            recipient=message.recipient,
                            addressee=message.addressee,
                            content=message.content,
                            attachment=message.attachment,
                        )
                        for message in messages_group.messages
                    )
                )

            case AgentMessageDraft() as message:
                workflow.send(
                    AgentMessage(
                        sender=self,
                        recipient=message.recipient,
                        addressee=message.addressee,
                        content=message.content,
                        attachment=message.attachment,
                    )
                )

            case result:
                workflow.finish_with(result)


type WorkflowOutput[Result] = frozenlist[AgentMessage] | AgentMessage | Result


@runtime_checkable
class WorkflowInvocation[Result](Protocol):
    async def __call__(
        self,
        workflow: "AgentWorkflow[Result]",
        message: AgentMessage,
    ) -> WorkflowOutput[Result]: ...


@final
class AgentWorkflow[Result]:
    def __init__(
        self,
        invocation: WorkflowInvocation[Result],
    ) -> None:
        self.identifier: UUID = uuid4()
        self._invocation: WorkflowInvocation[Result] = invocation
        self._runners: dict[UUID, WorkflowAgentRunner[Result]] = {}
        self._messages: AsyncStream[AgentMessage[Workflow]] = AsyncStream()
        self._message_trace: dict[UUID, Event] = {}
        self._result: Future[Workflow] = Future()

        freeze(self)

    def __del__(self) -> None:
        self.fail_with(CancelledError())

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

    def fail_with(
        self,
        result: BaseException,
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

    async def __call__(
        self,
        messages: AgentMessage,
        /,
        *_messages: AgentMessage,
        timeout: float = 600,  # default timeout is 10 minutes
    ) -> Result:
        async with ctx.nested(f"Workflow|{workflow.identifier}"):
            workflow.send(
                messages.updated(sender=WORKFLOW_ENTRY),
                *(message.updated(sender=WORKFLOW_ENTRY) for message in _messages),
            )

            return await workflow.execute(timeout=timeout)


@final
class WorkflowRunner[WorkflowResult]:
    @classmethod
    def spawn(
        cls,
        agent: AgentID,
        /,
        workflow: "AgentWorkflow[WorkflowResult]",
    ) -> Self: ...


def workflow[Result](
    invocation: WorkflowInvocation[Result],
) -> AgentWorkflow[Result]:
    def wrap(
        invocation: WorkflowInvocation[Result],
    ) -> AgentWorkflow[Result]:
        assert isfunction(invocation), "AgentWorkflow has to be defined from a function"  # nosec: B101

        return AgentWorkflow[Result](invocation=invocation)

    return wrap(invocation)
