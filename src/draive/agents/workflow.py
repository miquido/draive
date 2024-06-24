from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Future,
    Lock,
    TimerHandle,
    get_running_loop,
)
from typing import Any, Final, Self, cast, final
from uuid import UUID, uuid4

from draive.agents.runner import AgentRunner
from draive.agents.types import (
    AgentBase,
    AgentMessage,
    AgentMessageDraft,
    AgentWorkflowCurrent,
    AgentWorkflowStateAccess,
    WorkflowAgentBase,
)
from draive.parameters import ParameterPath, ParametrizedData
from draive.scope import ctx
from draive.types import MultimodalContent, MultimodalContentConvertible
from draive.utils import AsyncStream, freeze

__all__ = [
    "AgentWorkflow",
]


@final
class AgentWorkflow[WorkflowState: ParametrizedData, WorkflowResult]:
    @classmethod
    async def run(
        cls,
        messages: AgentMessageDraft,
        /,
        *__messages: AgentMessageDraft,
        state: WorkflowState,
        timeout: float | None = None,
    ) -> WorkflowResult:
        workflow: Self = cls(
            state=state,
            timeout=timeout,
        )
        async with ctx.nested(f"Workflow|{workflow.identifier}"):
            workflow.send(
                AgentMessage(
                    sender=WORKFLOW_ENTRY,
                    recipient=messages.recipient,
                    addressee=messages.addressee,
                    content=messages.content,
                ),
                *(
                    AgentMessage(
                        sender=WORKFLOW_ENTRY,
                        recipient=message.recipient,
                        addressee=message.addressee,
                        content=message.content,
                    )
                    for message in __messages
                ),
            )

            return await workflow.execute()

    def __init__(
        self,
        state: WorkflowState,
        timeout: float | None = None,
    ) -> None:
        current_state: WorkflowState = state

        def state_read() -> WorkflowState:
            return current_state

        def state_update(**parameters: Any) -> None:
            nonlocal current_state
            current_state = current_state.updated(**parameters)

        # TODO: prepare property based slice access
        self._workflow_state: AgentWorkflowStateAccess[WorkflowState] = AgentWorkflowStateAccess(
            lock=Lock(),
            read=state_read,
            update=state_update,
        )
        self.identifier: UUID = uuid4()
        self._runners: dict[UUID, AgentRunner] = {}
        self._messages: AsyncStream[AgentMessage] = AsyncStream()
        self._result: Future[WorkflowResult] = Future()
        self._timeout: float = timeout or 600  # default timeout is 10 minutes

        self._workflow_current: AgentWorkflowCurrent[WorkflowState, WorkflowResult] = (
            AgentWorkflowCurrent(
                access=self.state,
                send=self.send,
                finish=self.finish_with,
            )
        )

        freeze(self)

    def __del__(self) -> None:
        self.finish_with(CancelledError())

    def state[Parameter: ParametrizedData](
        self,
        path: ParameterPath[WorkflowState, Parameter] | Parameter | None,
        /,
    ) -> AgentWorkflowStateAccess[Parameter]:
        if path is not None:
            # TODO: prepare property based slice access
            raise NotImplementedError("Not implemented yet")

        return cast(AgentWorkflowStateAccess[Parameter], self._workflow_state)

    async def _deliver(
        self,
        message: AgentMessage,
        /,
    ) -> None:
        assert message.recipient.identifier != WORKFLOW_ENTRY.identifier  # nosec: B101

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

        runner: AgentRunner
        if running := self._runners.get(message.recipient.identifier):
            runner = running

        else:
            runner = AgentRunner.spawn(
                cast(WorkflowAgentBase[WorkflowState, WorkflowResult], message.recipient),
                workflow=self._workflow_current,
            )
            self._runners[message.recipient.identifier] = runner

        runner.send(message)

    def send(
        self,
        messages: AgentMessage,
        *_messages: AgentMessage,
    ) -> None:
        if self._result.done():
            return ctx.log_debug("Ignoring messages - workflow finished")

        self._messages.send(messages, *_messages)

    def finish_with(
        self,
        result: WorkflowResult | BaseException,
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

    async def execute(self) -> WorkflowResult:
        if self._result.done():
            raise RuntimeError("AgentWorkflow can be executed only once!")

        loop: AbstractEventLoop = get_running_loop()

        def on_timeout(
            future: Future[WorkflowResult],
        ) -> None:
            if future.done():
                return  # ignore if already finished

            # result future on its completion will ensure that task will complete
            future.set_exception(TimeoutError())

        timeout_handle: TimerHandle = loop.call_later(
            self._timeout,
            on_timeout,
            self._result,
        )

        def on_result(
            future: Future[WorkflowResult],
        ) -> None:
            timeout_handle.cancel()  # at this stage we no longer need timeout to trigger
            self._messages.finish()

        self._result.add_done_callback(on_result)

        # TODO: detect idle state - no agent working
        async for message in self._messages:
            await self._deliver(message)

        return await self._result


@final
class PlaceholderAgent(AgentBase):
    def __init__(self) -> None:
        self._identifier: UUID = UUID(int=0)

        freeze(self)

    @property
    def identifier(self) -> UUID:
        return self._identifier

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        *,
        addressee: AgentBase | None = None,
    ) -> AgentMessageDraft:
        raise RuntimeError("Can't address a message to a placeholder agent!")


WORKFLOW_ENTRY: Final[PlaceholderAgent] = PlaceholderAgent()
