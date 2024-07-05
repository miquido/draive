from asyncio import AbstractEventLoop, CancelledError, TimerHandle, gather, get_running_loop
from inspect import isfunction
from typing import Protocol, cast, final, overload, runtime_checkable
from uuid import UUID, uuid4

from draive.agents.definition import AgentMessage, AgentNode
from draive.agents.node import Agent, AgentError, AgentOutput
from draive.agents.runner import AgentRunner
from draive.helpers import VolatileMemory
from draive.parameters import DataModel, ParametrizedData
from draive.scope import ctx
from draive.types import (
    BasicMemory,
    Memory,
    MultimodalContent,
    MultimodalContentConvertible,
    frozenlist,
)
from draive.utils import MISSING, AsyncQueue, Missing, freeze, not_missing

__all__ = [
    "workflow",
    "AgentWorkflow",
    "AgentWorkflowInvocation",
    "AgentWorkflowOutput",
    "AgentWorkflowInput",
]


type AgentWorkflowInput = AgentMessage | AgentError
type AgentWorkflowOutput[Result: ParametrizedData | str] = (
    frozenlist[AgentMessage] | AgentMessage | Result | None
)


@runtime_checkable
class AgentWorkflowStateInitializer[AgentWorkflowState](Protocol):
    def __call__(self) -> AgentWorkflowState: ...


@runtime_checkable
class AgentWorkflowInvocation[AgentWorkflowState, AgentWorkflowResult: ParametrizedData | str](
    Protocol
):
    async def __call__(
        self,
        memory: BasicMemory[AgentWorkflowState],
        input: AgentWorkflowInput,  # noqa: A002
    ) -> AgentWorkflowOutput[AgentWorkflowResult]: ...


@final
class AgentWorkflow[AgentWorkflowState, AgentWorkflowResult: ParametrizedData | str]:
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

        freeze(self)

    async def __call__(
        self,
        memory: BasicMemory[AgentWorkflowState],
        input: AgentWorkflowInput,  # noqa: A002
    ) -> AgentWorkflowOutput[AgentWorkflowResult]:
        return await self._invocation(
            memory=memory,
            input=input,
        )

    async def run(
        self,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        state: AgentWorkflowState | None = None,
        timeout: float = 120,  # default timeout is 2 minutes
    ) -> AgentWorkflowResult:
        return await WorkflowRunner[AgentWorkflowState, AgentWorkflowResult].run(
            self,
            input=input,
            memory=VolatileMemory(state or self._state_initializer()),
            timeout=timeout,
        )

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
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
    def __call__[AgentWorkflowResult: ParametrizedData | str](
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

    def wrap[AgentWorkflowResult: ParametrizedData | str](
        invocation: AgentWorkflowInvocation[AgentWorkflowState, AgentWorkflowResult],
    ) -> AgentWorkflow[AgentWorkflowState, AgentWorkflowResult]:
        assert isfunction(invocation), "workflow has to be defined from a function"  # nosec: B101

        workflow_node: AgentNode = node or AgentNode(
            name=name or invocation.__qualname__,
            description=description or "",
        )

        def initialize_agent() -> Agent:
            agent_memory: BasicMemory[AgentWorkflowState] = cast(
                BasicMemory[AgentWorkflowState], VolatileMemory(state())
            )

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
        workflow_node._associate(  # pyright: ignore[reportPrivateUsage]
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
class WorkflowRunnerResultOutput[WorkflowResult: ParametrizedData | str](Protocol):
    def __call__(
        self,
        result: WorkflowResult,
        /,
    ) -> None: ...


class WorkflowHistory(DataModel):
    history: frozenlist[AgentMessage]


class WorkflowRunner[WorkflowState, WorkflowResult: ParametrizedData | str]:
    @classmethod
    async def run(
        cls,
        workflow: AgentWorkflow[WorkflowState, WorkflowResult],
        /,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
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
        self._workflow_queue: AsyncQueue[AgentMessage | AgentError] = AsyncQueue()
        self._workflow_memory: Memory[WorkflowState, WorkflowState] = memory
        self._workflow: AgentWorkflow[WorkflowState, WorkflowResult] = workflow
        self._agent_runners: dict[UUID, AgentRunner] = {}
        self._result: WorkflowResult | BaseException | Missing = MISSING

    def __del__(self) -> None:
        self.finish(result=CancelledError())

    @overload
    def send(
        self,
        error: AgentError,
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
        messages: AgentMessage | AgentError,
        /,
        *_messages: AgentMessage,
    ) -> None:
        pending: list[AgentMessage | AgentError] = [messages, *_messages]
        self._history.extend([message for message in pending if isinstance(message, AgentMessage)])
        self._workflow_queue.enqueue(*pending)

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

    async def wait(self) -> None:
        await gather(
            *[runner.wait() for runner in self._agent_runners.values()],
            return_exceptions=False,
        )

        await self._workflow_queue.wait()

    async def finalize(
        self,
        result: WorkflowResult | BaseException,
    ) -> None:
        self.finish(result=result)
        await self.wait()

    async def execute(
        self,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        timeout: float,
    ) -> WorkflowResult:
        assert not self._agent_runners, "WorkflowRunner can run only once!"  # nosec: B101

        if self._workflow_queue.finished or not_missing(self._result):
            raise RuntimeError("WorkflowRunner can run only once!")

        async with ctx.nested(self._workflow.node.__str__()):

            def on_timeout() -> None:
                self.finish(result=TimeoutError())

            timeout_handle: TimerHandle = self._loop.call_later(
                delay=timeout,
                callback=on_timeout,
            )

            self.send(self._workflow.address(input))

            async for element in self._workflow_queue:
                match element:
                    case AgentMessage() as message:
                        if message.recipient.identifier == self._workflow.node.identifier:
                            await self._handle(message)

                        elif runner := self._agent_runners.get(message.recipient.identifier):
                            runner.send(message)

                        else:
                            spawned_runner: AgentRunner = AgentRunner.run(
                                message.recipient,
                                output=self.send,
                            )
                            self._agent_runners[message.recipient.identifier] = spawned_runner
                            spawned_runner.send(message)

                    case error:
                        await self._handle(error)

            await self.wait()  # wait for completion of all runners

            timeout_handle.cancel()  # cancel the timeout

            ctx.record(WorkflowHistory(history=tuple(self._history)))

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
                    self.send(
                        *[message.updated(sender=self._workflow.node) for message in messages]
                    )

                case AgentMessage() as message:
                    self.send(message.updated(sender=self._workflow.node))

                case result:
                    self.finish(result=cast(WorkflowResult, result))

        except BaseException as exc:
            self.finish(result=exc)
