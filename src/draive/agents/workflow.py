from asyncio import AbstractEventLoop, CancelledError, Task, TimerHandle, gather, get_running_loop
from collections.abc import Sequence
from inspect import isfunction
from typing import Protocol, cast, final, overload, runtime_checkable
from uuid import UUID, uuid4

from haiway import MISSING, AsyncQueue, Missing, State, ctx, not_missing

from draive.agents.definition import AgentMessage, AgentNode
from draive.agents.errors import AgentException
from draive.agents.idle import IdleMonitor
from draive.agents.node import Agent, AgentOutput
from draive.agents.runner import AgentRunner
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)
from draive.parameters import DataModel
from draive.utils import Memory

__all__ = (
    "AgentWorkflow",
    "AgentWorkflowIdle",
    "AgentWorkflowInput",
    "AgentWorkflowInvocation",
    "AgentWorkflowOutput",
    "workflow",
)


type AgentWorkflowInput = AgentMessage | AgentException
type AgentWorkflowOutput[Result: DataModel | State | str] = (
    Sequence[AgentMessage] | AgentMessage | Result | None
)


@runtime_checkable
class AgentWorkflowStateInitializer[AgentWorkflowState](Protocol):
    def __call__(self) -> AgentWorkflowState: ...


@runtime_checkable
class AgentWorkflowInvocation[AgentWorkflowState, AgentWorkflowResult: DataModel | State | str](
    Protocol
):
    async def __call__(
        self,
        memory: Memory[AgentWorkflowState, AgentWorkflowState],
        input: AgentWorkflowInput,  # noqa: A002
    ) -> AgentWorkflowOutput[AgentWorkflowResult]: ...


class AgentWorkflowIdle(AgentException):
    pass


@final
class AgentWorkflow[AgentWorkflowState, AgentWorkflowResult: DataModel | State | str]:
    def __init__(
        self,
        node: AgentNode,
        invocation: AgentWorkflowInvocation[
            AgentWorkflowState,
            AgentWorkflowResult,
        ],
        state_initializer: AgentWorkflowStateInitializer[AgentWorkflowState],
    ) -> None:
        self.node: AgentNode = node
        self._invocation: AgentWorkflowInvocation[
            AgentWorkflowState,
            AgentWorkflowResult,
        ] = invocation
        self._state_initializer: AgentWorkflowStateInitializer[AgentWorkflowState] = (
            state_initializer
        )

    async def __call__(
        self,
        memory: Memory[AgentWorkflowState, AgentWorkflowState],
        input: AgentWorkflowInput,  # noqa: A002
    ) -> AgentWorkflowOutput[AgentWorkflowResult]:
        return await self._invocation(
            memory=memory,
            input=input,
        )

    async def run(
        self,
        input: Multimodal,  # noqa: A002
        state: AgentWorkflowState | None = None,
        timeout: float = 120,  # default timeout is 2 minutes
    ) -> AgentWorkflowResult:
        return await WorkflowRunner[AgentWorkflowState, AgentWorkflowResult].run(
            self,
            input=input,
            memory=Memory[AgentWorkflowState, AgentWorkflowState].volatile(
                initial=state or self._state_initializer()
            ),
            timeout=timeout,
        )

    def address(
        self,
        content: Multimodal,
        /,
        *_content: Multimodal,
        addressee: AgentNode | None = None,
    ) -> AgentMessage:
        return AgentMessage(
            identifier=uuid4(),
            sender=MISSING,
            recipient=self.node,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
            responding=None,
        )


class PartialAgentWorkflowWrapper[AgentWorkflowState](Protocol):
    def __call__[AgentWorkflowResult: DataModel | State | str](
        self,
        invocation: AgentWorkflowInvocation[AgentWorkflowState, AgentWorkflowResult],
    ) -> AgentWorkflow[AgentWorkflowState, AgentWorkflowResult]: ...


@overload
def workflow[AgentWorkflowState](
    node: AgentNode,
    /,
    *,
    state: AgentWorkflowStateInitializer[AgentWorkflowState],
) -> PartialAgentWorkflowWrapper[AgentWorkflowState]: ...


@overload
def workflow[AgentWorkflowState](
    *,
    name: str | None = None,
    description: str,
    state: AgentWorkflowStateInitializer[AgentWorkflowState],
) -> PartialAgentWorkflowWrapper[AgentWorkflowState]: ...


def workflow[AgentWorkflowState](
    node: AgentNode | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    state: AgentWorkflowStateInitializer[AgentWorkflowState],
) -> PartialAgentWorkflowWrapper[AgentWorkflowState]:
    assert node is None or (  # nosec: B101
        name is None and description is None
    ), "Can't specify both agent node and name/description"
    assert (  # nosec: B101
        description is not None or node is not None
    ), "Either agent node or description has to be provided"

    def wrap[AgentWorkflowResult: DataModel | State | str](
        invocation: AgentWorkflowInvocation[AgentWorkflowState, AgentWorkflowResult],
    ) -> AgentWorkflow[AgentWorkflowState, AgentWorkflowResult]:
        assert isfunction(invocation), "workflow has to be defined from a function"  # nosec: B101

        workflow_node: AgentNode = node or AgentNode(
            name=name or invocation.__qualname__,
            description=description or "",
        )

        def initialize_agent() -> Agent:
            agent_memory: Memory[AgentWorkflowState, AgentWorkflowState] = Memory[
                AgentWorkflowState, AgentWorkflowState
            ].volatile(initial=state())

            async def agent(message: AgentMessage) -> AgentOutput:
                match await cast(  # pyright fails to properly type this function
                    AgentWorkflowInvocation[AgentWorkflowState, AgentWorkflowResult],
                    invocation,
                )(
                    memory=agent_memory,
                    input=message,
                ):
                    case None:
                        pass

                    case tuple() as messages:
                        return messages

                    case AgentMessage() as message:
                        return message

                    case result:
                        return message.respond(
                            MultimodalContent.of(
                                result if isinstance(result, DataModel) else str(result)
                            )
                        )

            return agent

        # workflow can run as an agent with limited capabilities
        workflow_node.associate(
            initialize_agent,
            concurrent=False,  # workflow can't be concurrent
        )

        return AgentWorkflow[AgentWorkflowState, AgentWorkflowResult](
            node=workflow_node,
            invocation=invocation,
            state_initializer=state,
        )

    return wrap


@runtime_checkable
class WorkflowRunnerOutput(Protocol):
    def __call__(
        self,
        messages: AgentMessage,
        /,
        *_messages: AgentMessage,
    ) -> None: ...


@runtime_checkable
class WorkflowRunnerResultOutput[WorkflowResult: DataModel | State | str](Protocol):
    def __call__(
        self,
        result: WorkflowResult,
        /,
    ) -> None: ...


class WorkflowRunner[WorkflowState, WorkflowResult: DataModel | State | str]:
    @classmethod
    async def run(
        cls,
        workflow: AgentWorkflow[WorkflowState, WorkflowResult],
        /,
        input: Multimodal,  # noqa: A002
        memory: Memory[WorkflowState, WorkflowState],
        timeout: float,
    ) -> WorkflowResult:
        return await cls(
            workflow=workflow,
            memory=memory,
            loop=get_running_loop(),
        ).execute(
            input=input,
            timeout=timeout,
        )

    def __init__(
        self,
        workflow: AgentWorkflow[WorkflowState, WorkflowResult],
        memory: Memory[WorkflowState, WorkflowState],
        loop: AbstractEventLoop,
    ) -> None:
        self._loop: AbstractEventLoop = loop
        self._history: list[AgentMessage] = []
        self._workflow_queue: AsyncQueue[AgentMessage | AgentException] = AsyncQueue()
        self._workflow_memory: Memory[WorkflowState, WorkflowState] = memory
        self._workflow: AgentWorkflow[WorkflowState, WorkflowResult] = workflow
        self._agent_runners: dict[UUID, AgentRunner] = {}
        self._status: IdleMonitor = IdleMonitor()
        self._result: WorkflowResult | BaseException | Missing = MISSING

    def __del__(self) -> None:
        self.finish(result=CancelledError())

    @overload
    def send(
        self,
        error: AgentException,
        /,
    ) -> None: ...

    @overload
    def send(
        self,
        messages: AgentMessage,
        /,
        *_messages: AgentMessage,
    ) -> None: ...

    def send(
        self,
        messages: AgentMessage | AgentException,
        /,
        *_messages: AgentMessage,
    ) -> None:
        pending: list[AgentMessage | AgentException] = [messages, *_messages]
        for _ in pending:  # ensure not idle when received a message
            self._status.enter_task()
        self._history.extend([message for message in pending if isinstance(message, AgentMessage)])
        self._workflow_queue.enqueue(*pending)

    @property
    def finished(self) -> bool:
        return not_missing(self._result)

    def finish(
        self,
        result: WorkflowResult | BaseException,
    ) -> None:
        if not_missing(self._result):
            return  # already finished

        ctx.log_debug("Finishing workflow with result: %s", result)

        self._result = result

        for runner in self._agent_runners.values():
            runner.finish()

        self._workflow_queue.finish()

    async def _wait(self) -> None:
        await gather(
            *[runner.wait() for runner in self._agent_runners.values()],
            return_exceptions=False,
        )

    async def finalize(
        self,
        result: WorkflowResult | BaseException,
    ) -> None:
        self.finish(result=result)
        await self._wait()

    async def execute(  # noqa: PLR0912
        self,
        input: Multimodal,  # noqa: A002
        timeout: float,
    ) -> WorkflowResult:
        assert not self._agent_runners, "WorkflowRunner can run only once!"  # nosec: B101

        if self._workflow_queue.is_finished or not_missing(self._result):
            raise RuntimeError("WorkflowRunner can run only once!")

        async with ctx.scope(self._workflow.node.__str__()):

            def on_timeout() -> None:
                self.finish(result=TimeoutError())

            timeout_handle: TimerHandle = self._loop.call_later(
                delay=timeout,
                callback=on_timeout,
            )

            self.send(self._workflow.address(input))

            idle_monitor_task: Task[None] = ctx.spawn(self._idle_monitor)

            try:
                async for element in self._workflow_queue:
                    try:
                        match element:
                            case AgentMessage() as message:
                                if message.recipient.identifier == self._workflow.node.identifier:
                                    await self._handle(message)

                                elif runner := self._agent_runners.get(
                                    message.recipient.identifier
                                ):
                                    runner.send(message)

                                else:
                                    spawned_runner: AgentRunner = AgentRunner.run(
                                        message.recipient,
                                        output=self.send,
                                        status=self._status.nested(),
                                    )
                                    self._agent_runners[message.recipient.identifier] = (
                                        spawned_runner
                                    )
                                    spawned_runner.send(message)

                            case error:
                                await self._handle(error)

                    finally:
                        self._status.exit_task()

            except CancelledError:
                pass  # just end on cancel

            finally:  # cleanup status when exiting
                timeout_handle.cancel()  # cancel the timeout
                idle_monitor_task.cancel()  # finish idle monitor
                self._status.exit_all()

            await self._wait()  # wait for completion of all runners

            match self._result:
                case BaseException() as exc:
                    raise exc

                case Missing():
                    raise RuntimeError("Invalid workflow state")

                case result:
                    return result

    async def _handle(
        self,
        input: AgentWorkflowInput,  # noqa: A002
        /,
    ) -> None:
        try:
            match await self._workflow(
                memory=self._workflow_memory,
                input=input,
            ):
                case None:
                    pass  # nothing to do

                case [*messages]:
                    self.send(*[message._with_sender(self._workflow.node) for message in messages])  # pyright: ignore[reportPrivateUsage]

                case AgentMessage() as message:
                    self.send(message._with_sender(self._workflow.node))  # pyright: ignore[reportPrivateUsage]

                case result:
                    self.finish(result=cast(WorkflowResult, result))

        except BaseException as exc:
            self.finish(result=exc)

    async def _idle_monitor(self) -> None:
        while not self.finished:  # run until finished
            await self._status.wait_idle()  # wait until all agents become idle

            if self.finished:  # check if that is not due to finishing
                return  # workflow might be finished at this stage

            ctx.log_warning("Detected workflow idle state, attempting to resume...")

            # when workflow should still be running (did not provided result yet)
            # but no agent is working then notify workflow node about the error
            self.send(AgentWorkflowIdle())
