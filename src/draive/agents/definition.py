from collections.abc import Callable
from inspect import isfunction
from typing import Literal, Protocol, cast, final, overload
from uuid import UUID, uuid4

from draive.agents.errors import AgentException
from draive.agents.types import (
    AgentBase,
    AgentCurrent,
    AgentInput,
    AgentInvocation,
    AgentMessage,
    AgentMessageDraft,
    AgentMessageDraftGroup,
    AgentOutput,
    AgentWorkflowCurrent,
    WorkflowAgentBase,
)
from draive.agents.workflow import AgentWorkflow
from draive.helpers import ConstantMemory, VolatileMemory
from draive.parameters import ParametrizedData, Stateless
from draive.scope import ctx
from draive.types import (
    Memory,
    MultimodalContent,
    MultimodalContentConvertible,
)
from draive.utils import freeze, mimic_function

__all__ = [
    "agent",
    "Agent",
]


@final
class Agent[
    AgentState,
    AgentStateScratch,
    WorkflowState: ParametrizedData,
    WorkflowResult,
](WorkflowAgentBase[WorkflowState, WorkflowResult]):
    def __init__(  # noqa: PLR0913
        self,
        identifier: UUID,
        name: str,
        description: str | None,
        memory: Callable[[], Memory[AgentState, AgentStateScratch]],
        invocation: AgentInvocation[
            AgentState,
            AgentStateScratch,
            WorkflowState,
            WorkflowResult,
        ],
        concurrent: bool,
    ) -> None:
        self._identifier: UUID = identifier
        self.name: str = name
        self.description: str | None = description
        self._memory: Callable[[], Memory[AgentState, AgentStateScratch]] = memory
        self._invocation: AgentInvocation[
            AgentState,
            AgentStateScratch,
            WorkflowState,
            WorkflowResult,
        ] = invocation
        self._concurrent: bool = concurrent

        mimic_function(invocation, within=self)
        freeze(self)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, Agent):
            return self._identifier == other._identifier
        else:
            return False

    def __hash__(self) -> int:
        return hash(self._identifier)

    def __str__(self) -> str:
        return f"Agent|{self.name}|{self._identifier}"

    async def __call__(
        self,
        current: AgentCurrent[
            AgentState,
            AgentState,
            WorkflowState,
            WorkflowResult,
        ],
        message: AgentMessage,
    ) -> AgentOutput[WorkflowResult]:
        with ctx.nested(
            f"{self.__str__()}|Invocation|{message.identifier}",
            metrics=[message],
        ):
            return await self._invocation(
                current,
                message,
            )

    async def _draive(  # yes, it is the name of the library
        self,
        input: AgentInput,  # noqa: A002
        workflow: AgentWorkflowCurrent[WorkflowState, WorkflowResult],
    ) -> None:
        with ctx.nested(self.__str__()):
            agent_current: AgentCurrent[
                AgentState, AgentStateScratch, WorkflowState, WorkflowResult
            ] = AgentCurrent(
                agent=self,
                memory=self._memory(),
                workflow=workflow,
            )
            try:
                if self._concurrent:
                    # process all messages concurrently by spawning a task for each
                    async for message in input:
                        ctx.spawn_subtask(
                            self._handle_message,
                            message,
                            agent_current,
                            workflow,
                        )

                else:
                    # process only one message at the time by waiting for results
                    async for message in input:
                        await self._handle_message(
                            message,
                            agent_current=agent_current,
                            workflow=workflow,
                        )

            except Exception as exc:
                workflow.finish_with(AgentException(cause=exc))

    async def _handle_message(
        self,
        message: AgentMessage,
        /,
        agent_current: AgentCurrent[AgentState, AgentStateScratch, WorkflowState, WorkflowResult],
        workflow: AgentWorkflowCurrent[WorkflowState, WorkflowResult],
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
                    )
                )

            case result:
                workflow.finish_with(result)

    @property
    def identifier(self) -> UUID:
        return self._identifier

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        *,
        addressee: AgentBase | None = None,
    ) -> AgentMessageDraft:
        return AgentMessageDraft(
            recipient=self,
            addressee=addressee,
            content=MultimodalContent.of(content),
        )

    async def start_workflow(
        self,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        /,
        state: WorkflowState,
        timeout: float | None = None,
    ) -> WorkflowResult:
        return await AgentWorkflow[WorkflowState, WorkflowResult].run(
            self.address(input, addressee=None),
            state=state,
            timeout=timeout,
        )


class StatelessAgentWrapper(Protocol):
    def __call__[WorkflowState: ParametrizedData, WorkflowResult](
        self,
        invocation: AgentInvocation[Stateless, Stateless, WorkflowState, WorkflowResult],
    ) -> Agent[Stateless, Stateless, WorkflowState, WorkflowResult]: ...


class PartialAgentWrapper[LocalState, LocalStateScratch](Protocol):
    def __call__[WorkflowState: ParametrizedData, WorkflowResult](
        self,
        invocation: AgentInvocation[LocalState, LocalStateScratch, WorkflowState, WorkflowResult],
    ) -> Agent[LocalState, LocalStateScratch, WorkflowState, WorkflowResult]: ...


@overload
def agent[WorkflowState: ParametrizedData, WorkflowResult](
    invocation: AgentInvocation[Stateless, Stateless, WorkflowState, WorkflowResult],
    /,
) -> Agent[Stateless, Stateless, WorkflowState, WorkflowResult]: ...


@overload
def agent(
    *,
    name: str | None = None,
    description: str | None = None,
    concurrent: Literal[True],
) -> StatelessAgentWrapper: ...


@overload
def agent[LocalState](
    *,
    name: str | None = None,
    description: str | None = None,
    initial_state: Callable[[], LocalState],
    concurrent: Literal[False] = False,
) -> PartialAgentWrapper[LocalState, LocalState]: ...


@overload
def agent[LocalState, LocalStateScratch](
    *,
    name: str | None = None,
    description: str | None = None,
    memory: Memory[LocalState, LocalStateScratch],
    concurrent: Literal[False] = False,
) -> PartialAgentWrapper[LocalState, LocalStateScratch]: ...


def agent[  # noqa: PLR0913
    LocalState,
    LocalStateScratch,
    WorkflowState: ParametrizedData,
    WorkflowResult,
](
    invocation: AgentInvocation[LocalState, LocalStateScratch, WorkflowState, WorkflowResult]
    | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    initial_state: Callable[[], LocalState] | None = None,
    memory: Memory[LocalState, LocalStateScratch] | None = None,
    concurrent: bool = False,
) -> (
    StatelessAgentWrapper
    | PartialAgentWrapper[LocalState, LocalStateScratch]
    | Agent[LocalState, LocalStateScratch, WorkflowState, WorkflowResult]
):
    assert initial_state is None or memory is None, "Can't specify both initial state and memory"  # nosec: B101

    def agent_memory() -> Memory[LocalState, LocalStateScratch]:
        if memory is not None:
            return memory

        elif initial_state is not None:
            return cast(Memory[LocalState, LocalStateScratch], VolatileMemory(initial_state()))

        else:
            return cast(Memory[LocalState, LocalStateScratch], ConstantMemory(Stateless()))

    def wrap[WrappedWorkflowState: ParametrizedData, WrappedWorkflowResult](
        invocation: AgentInvocation[
            LocalState, LocalStateScratch, WrappedWorkflowState, WrappedWorkflowResult
        ],
    ) -> Agent[LocalState, LocalStateScratch, WrappedWorkflowState, WrappedWorkflowResult]:
        assert isfunction(invocation), "Agent has to be defined from a function"  # nosec: B101

        return Agent(
            identifier=uuid4(),
            name=name or invocation.__qualname__,
            description=description,
            memory=agent_memory,
            invocation=invocation,
            concurrent=concurrent,
        )

    if invocation := invocation:
        return wrap(invocation)

    else:
        return wrap
