from abc import ABC, abstractmethod
from asyncio import Event, Lock
from types import TracebackType
from typing import Any, Protocol, cast, runtime_checkable
from uuid import UUID, uuid4

from draive.parameters import ParameterPath, ParametrizedData
from draive.types import Memory, MultimodalContent, MultimodalContentConvertible
from draive.utils import MISSING, AsyncStream, Missing, freeze, is_missing, not_missing

__all__ = [
    "AgentID",
    "AgentInput",
    "AgentOutput",
    "AgentMessage",
    "AgentWorkflowBase",
    "WorkflowState",
]


class AgentID[Workflow]:
    def __init__(
        self,
        workflow: type[Workflow],
        /,
        identifier: UUID | None = None,
    ) -> None:
        self._identifier: UUID = identifier or uuid4()
        self._agent: AgentBase[Workflow] | Missing

    @property
    def identifier(self) -> UUID:
        return self._identifier

    @property
    def name(self) -> str:
        assert not_missing(self._agent), "AgentID has to be associated with an agent to be used!"  # nosec: B101
        return self._agent.name

    @property
    def description(self) -> str:
        assert not_missing(self._agent), "AgentID has to be associated with an agent to be used!"  # nosec: B101
        return self._agent.description

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if isinstance(other, AgentID):
            return self._identifier == other._identifier
        else:
            return False

    def _associate(
        self,
        agent: Any,
    ) -> None:
        assert isinstance(agent, AgentBase)  # nosec: B101
        assert is_missing(self._agent), "AgentID can be associated with only one agent"  # nosec: B101
        self._agent = agent
        freeze(self)

    async def _draive(  # yes, it is the name of the library
        self,
        input: "AgentInput[Workflow]",  # noqa: A002
        /,
        workflow: "AgentWorkflowBase[Workflow]",
    ) -> None:
        assert not_missing(self._agent), "AgentID has to be associated with an agent to be used!"  # nosec: B101
        await self._agent._draive(input, workflow=workflow)  # pyright: ignore[reportPrivateUsage]

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        addressee: "AgentID[Workflow] | None" = None,
    ) -> "AgentMessage[Workflow]":
        assert not_missing(self._agent), "AgentID has to be associated with an agent to be used!"  # nosec: B101
        return AgentMessage(
            identifier=uuid4(),
            completion=Event(),
            sender=MISSING,
            recipient=self,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
        )


class AgentMessage[Workflow](ParametrizedData):
    identifier: UUID
    completion: Event
    sender: AgentID[Workflow] | Missing
    recipient: AgentID[Workflow]
    addressee: AgentID[Workflow] | None
    content: MultimodalContent

    @property
    def should_respond(self) -> bool:
        return self.addressee is not None

    def response(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        addressee: AgentID[Workflow] | None = None,
    ) -> "AgentMessage[Workflow]":
        assert not_missing(self.sender), "Missing message sender, can't respond to message drafts!"  # nosec: B101
        return AgentMessage(
            identifier=uuid4(),
            completion=Event(),
            sender=MISSING,
            recipient=self.addressee or self.sender,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
        )


type AgentInput[Workflow] = AsyncStream[AgentMessage[Workflow]]
type AgentOutput[Workflow] = (
    tuple[AgentMessage[Workflow], ...] | AgentMessage[Workflow] | Workflow | None
)


class WorkflowState:
    def __init__(
        self,
        initial: tuple[ParametrizedData, ...] | ParametrizedData,
        /,
    ) -> None:
        self._lock: Lock = Lock()
        self._state: dict[type, ParametrizedData]
        match initial:
            case [*elements]:
                self._state = {type(element): element for element in elements}

            case element:
                self._state = {type(element): element}

        freeze(self)

    def state[Root: ParametrizedData, Parameter](
        self,
        path: ParameterPath[Root, Parameter] | Parameter,
        /,
    ) -> Parameter | Missing:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"
        assert self._lock.locked(), "Can't access workflow state without lock!"  # nosec: B101
        root: Root | Missing = cast(
            Root | Missing,
            self._state.get(
                cast(ParameterPath[Root, Parameter], path).__root__,
                MISSING,
            ),
        )

        if not_missing(root):
            try:
                return cast(ParameterPath[Root, Parameter], path)(root)

            except (AttributeError, KeyError, IndexError):
                return MISSING

        else:
            return MISSING

    def update[Root: ParametrizedData, Parameter](
        self,
        path: ParameterPath[Root, Parameter] | Parameter,
        /,
        value: Parameter,
    ) -> bool:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"
        assert self._lock.locked(), "Can't access workflow state without lock!"
        root: Root | Missing = cast(
            Root | Missing,
            self._state.get(
                cast(ParameterPath[Root, Parameter], path).__root__,
                MISSING,
            ),
        )

        if not_missing(root):
            try:
                cast(ParameterPath[Root, Parameter], path)(root, updated=value)
                return True

            except (AttributeError, KeyError, IndexError):
                return False

        elif not path.__components__ and isinstance(
            value, cast(ParameterPath[Root, Parameter], path).__root__
        ):
            self._state[cast(ParameterPath[Root, Parameter], path).__root__] = value
            return True

        else:
            return False

    async def __aenter__(self) -> None:
        await self._lock.__aenter__()

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


class AgentWorkflowBase[Workflow](ABC):
    def __init__(
        self,
        state: WorkflowState,
    ) -> None:
        self._state: WorkflowState = state
        self._messages: AsyncStream[AgentMessage[Workflow]] = AsyncStream()

    @property
    def access(self) -> WorkflowState:
        return self._state

    async def state[Root: ParametrizedData, Parameter](
        self,
        path: ParameterPath[Root, Parameter] | Parameter,
        /,
    ) -> Parameter | Missing:
        async with self._state:
            return self._state.state(path)

    async def update[Root: ParametrizedData, Parameter](
        self,
        path: ParameterPath[Root, Parameter] | Parameter,
        /,
        value: Parameter,
    ) -> bool:
        async with self._state:
            return self._state.update(
                path,
                value=value,
            )

    def send(
        self,
        messages: AgentMessage[Workflow],
        /,
        *_messages: AgentMessage[Workflow],
    ) -> None:
        self._messages.send(messages, *_messages)

    @abstractmethod
    def finish_with(
        self,
        result: Workflow | BaseException,
        /,
    ) -> None: ...


class AgentBase[Workflow]:
    def __init__(
        self,
        agent_id: AgentID[Workflow],
        name: str,
        description: str,
    ) -> None:
        self._agent_id: AgentID[Workflow] = agent_id
        self.name: str = name
        self.description: str = description

        self._agent_id._associate(agent=self)  # pyright: ignore[reportPrivateUsage]

    @property
    def identifier(self) -> UUID:
        return self._agent_id.identifier

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        addressee: AgentID[Workflow] | None = None,
    ) -> AgentMessage[Workflow]:
        return self._agent_id.address(
            content,
            *_content,
            addressee=addressee,
        )

    @abstractmethod
    async def _draive(  # yes, it is the name of the library
        self,
        input: AgentInput[Workflow],  # noqa: A002
        /,
        workflow: AgentWorkflowBase[Workflow],
    ) -> None: ...


@runtime_checkable
class AgentInvocation[AgentState, AgentStateScratch, Workflow](Protocol):
    async def __call__(
        self,
        workflow: AgentWorkflowBase[Workflow],
        memory: Memory[AgentState, AgentStateScratch],
        message: AgentMessage[Workflow],
    ) -> AgentOutput[Workflow]: ...


@runtime_checkable
class StatelessAgentInvocation[Workflow](Protocol):
    async def __call__(
        self,
        workflow: AgentWorkflowBase[Workflow],
        message: AgentMessage[Workflow],
    ) -> AgentOutput[Workflow]: ...
