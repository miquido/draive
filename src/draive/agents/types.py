from abc import ABC, abstractmethod
from asyncio import Lock
from types import TracebackType
from typing import Any, Protocol, Self, final, overload, runtime_checkable
from uuid import UUID, uuid4

from draive.parameters import DataModel, Field, ParameterPath, ParametrizedData, Stateless
from draive.types import Memory, MultimodalContent, MultimodalContentConvertible, frozenlist
from draive.utils import AsyncStream, freeze

__all__ = [
    "AgentBase",
    "AgentInput",
    "AgentOutput",
    "AgentMessage",
    "AgentMessageDraft",
    "AgentCurrent",
    "StatelessAgentCurrent",
    "AgentMessageDraftGroup",
]


class AgentWorkflowStateRead[WorkflowState: ParametrizedData](Protocol):
    def __call__(self) -> WorkflowState: ...


class AgentWorkflowStateUpdate(Protocol):
    def __call__(
        self,
        **parameters: Any,
    ) -> None: ...


@final
class AgentWorkflowStateAccess[WorkflowState: ParametrizedData]:
    def __init__(
        self,
        lock: Lock,
        read: AgentWorkflowStateRead[WorkflowState],
        update: AgentWorkflowStateUpdate,
    ) -> None:
        self._lock: Lock = lock
        self._read: AgentWorkflowStateRead[WorkflowState] = read
        self._update: AgentWorkflowStateUpdate = update

        freeze(self)

    @property
    def current(self) -> WorkflowState:
        assert self._lock.locked(), "Can't access workflow state out of lock!"  # nosec: B101
        return self._read()

    def update(
        self,
        /,
        **parameters: Any,
    ) -> None:
        assert self._lock.locked(), "Can't access workflow state out of lock!"  # nosec: B101
        self._update(**parameters)

    async def __aenter__(self) -> Self:
        await self._lock.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._lock.__aexit__(
            exc_type,
            exc_val,
            exc_tb,
        )


class AgentBase(ABC):
    @property
    @abstractmethod
    def identifier(self) -> UUID: ...

    @abstractmethod
    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        *,
        addressee: "AgentBase | None" = None,
    ) -> "AgentMessageDraft": ...


class AgentMessageDraft(DataModel):
    recipient: AgentBase
    addressee: AgentBase | None
    content: MultimodalContent


class AgentMessageDraftGroup:
    def __init__(
        self,
        messages: AgentMessageDraft,
        /,
        *__messages: AgentMessageDraft,
    ) -> None:
        self.messages: frozenlist[AgentMessageDraft] = (messages, *__messages)

        freeze(self)


class AgentMessage(DataModel):
    identifier: UUID = Field(default_factory=lambda: uuid4())
    sender: AgentBase
    recipient: AgentBase
    addressee: AgentBase | None
    content: MultimodalContent

    @property
    def should_respond(self) -> bool:
        return self.addressee is not None

    def response(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        *,
        addressee: AgentBase | None = None,
    ) -> AgentMessageDraft:
        return AgentMessageDraft(
            recipient=self.addressee or self.sender,
            addressee=addressee,
            content=MultimodalContent.of(content),
        )


type AgentInput = AsyncStream[AgentMessage]
type AgentOutput[WorkflowResult] = (
    AgentMessageDraftGroup | AgentMessageDraft | WorkflowResult | None
)


class WorkflowAgentBase[WorkflowState: ParametrizedData, WorkflowResult](AgentBase):
    @abstractmethod
    async def _draive(  # yes, it is the name of the library
        self,
        input: AgentInput,  # noqa: A002
        workflow: "AgentWorkflowCurrent[WorkflowState, WorkflowResult]",
    ) -> None: ...

    @abstractmethod
    async def start_workflow(
        self,
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        /,
        state: WorkflowState,
        timeout: float | None = None,
    ) -> WorkflowResult: ...


class WorkflowStateAccess[WorkflowState: ParametrizedData](Protocol):
    def __call__[Parameter: ParametrizedData](
        self,
        path: ParameterPath[WorkflowState, Parameter] | Parameter | None,
        /,
    ) -> AgentWorkflowStateAccess[Parameter]: ...


class WorkflowMessageSend(Protocol):
    def __call__(
        self,
        *messages: AgentMessage,
    ) -> None: ...


class WorkflowFinish[WorkflowResult](Protocol):
    def __call__(
        self,
        result: WorkflowResult | BaseException,
        /,
    ) -> None: ...


@final
class AgentWorkflowCurrent[WorkflowState: ParametrizedData, WorkflowResult]:
    def __init__(
        self,
        access: WorkflowStateAccess[WorkflowState],
        send: WorkflowMessageSend,
        finish: WorkflowFinish[WorkflowResult],
    ) -> None:
        self._access: WorkflowStateAccess[WorkflowState] = access
        self._send: WorkflowMessageSend = send
        self._finish: WorkflowFinish[WorkflowResult] = finish

    @overload
    def state(
        self,
        /,
    ) -> AgentWorkflowStateAccess[WorkflowState]: ...

    @overload
    def state[Parameter: ParametrizedData](
        self,
        path: ParameterPath[WorkflowState, Parameter] | Parameter,
        /,
    ) -> AgentWorkflowStateAccess[Parameter]: ...

    def state[Parameter: ParametrizedData](
        self,
        path: ParameterPath[WorkflowState, Parameter] | Parameter | None = None,
        /,
    ) -> AgentWorkflowStateAccess[Parameter]:
        return self._access(path)

    async def update(
        self,
        /,
        **parameters: Any,
    ) -> None:
        async with self.state() as access:
            access.update(**parameters)

    def send(
        self,
        *messages: AgentMessage,
    ) -> None:
        self._send(*messages)

    def finish_with(
        self,
        result: WorkflowResult | BaseException,
        /,
    ) -> None:
        return self._finish(result)


@final
class AgentCurrent[
    AgentState,
    AgentStateScratch,
    WorkflowState: ParametrizedData,
    WorkflowResult,
]:
    def __init__(
        self,
        agent: AgentBase,
        memory: Memory[AgentState, AgentStateScratch],
        workflow: AgentWorkflowCurrent[WorkflowState, WorkflowResult],
    ) -> None:
        self._agent: AgentBase = agent
        self._memory: Memory[AgentState, AgentStateScratch] = memory
        self._workflow: AgentWorkflowCurrent[WorkflowState, WorkflowResult] = workflow

        freeze(self)

    @property
    async def state(self) -> AgentState:
        return await self._memory.recall()

    async def remember(
        self,
        *items: AgentStateScratch,
    ) -> None:
        await self._memory.remember(*items)

    @property
    def workflow(self) -> AgentWorkflowCurrent[WorkflowState, WorkflowResult]:
        return self._workflow

    def send(
        self,
        *messages: AgentMessageDraft,
    ) -> None:
        self._workflow.send(
            *(
                AgentMessage(
                    sender=self._agent,
                    recipient=message.recipient,
                    addressee=message.addressee,
                    content=message.content,
                )
                for message in messages
            ),
        )


type StatelessAgentCurrent[
    WorkflowState: ParametrizedData,
    WorkflowResult,
] = AgentCurrent[Stateless, Stateless, WorkflowState, WorkflowResult]


@runtime_checkable
class AgentInvocation[
    AgentState,
    AgentStateScratch,
    WorkflowState: ParametrizedData,
    WorkflowResult,
](Protocol):
    async def __call__(
        self,
        current: AgentCurrent[AgentState, AgentStateScratch, WorkflowState, WorkflowResult],
        message: AgentMessage,
    ) -> AgentOutput[WorkflowResult]: ...
