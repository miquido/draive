from inspect import isfunction
from typing import Any, Protocol, cast, final, overload, runtime_checkable
from uuid import UUID, uuid4

from draive.helpers import ConstantMemory, VolatileMemory
from draive.parameters import DataModel, ParametrizedData, Stateless
from draive.types import Memory, MultimodalContent, MultimodalContentConvertible, frozenlist
from draive.utils import MISSING, AsyncStream, Missing, freeze, is_missing, not_missing

__all__ = [
    "AgentID",
    "Agent",
    "AgentInput",
    "AgentOutput",
    "AgentMessage",
]


@final
class AgentID:
    def __init__(
        self,
        identifier: UUID | None = None,
        /,
        *,
        name: str,
        capabilities: str | None = None,
    ) -> None:
        self._identifier: UUID = identifier or uuid4()
        self._name: str = name
        self._capabilities: str | None = capabilities
        self._agent_ref: Agent[Any, Any] | Missing

    def __hash__(self) -> int:
        return hash(self._identifier)

    def __str__(self) -> str:
        return f"Agent|{self.name}|{self._identifier}"

    @property
    def identifier(self) -> UUID:
        return self._identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> str | None:
        return self._capabilities

    @property
    def _agent(self) -> "Agent[Any, Any]":
        assert not_missing(
            self._agent_ref
        ), "AgentID has to be associated with an agent to be used!"  # nosec: B101
        return self._agent_ref

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
        agent: "Agent[Any, Any]",
    ) -> None:
        assert isinstance(agent, Agent)  # nosec: B101
        assert is_missing(self._agent_ref), "AgentID can be associated with only one agent"  # nosec: B101
        self._agent_ref = agent
        freeze(self)

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        artifacts: frozenlist[DataModel] = (),
        addressee: "AgentID | None" = None,
    ) -> "AgentMessage":
        assert not_missing(
            self._agent_ref
        ), "AgentID has to be associated with an agent to be used!"  # nosec: B101
        return AgentMessage(
            identifier=uuid4(),
            sender=MISSING,
            recipient=self,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
            artifacts=artifacts,
        )


@final
class AgentMessage(ParametrizedData):
    identifier: UUID
    sender: AgentID | Missing
    recipient: AgentID
    addressee: AgentID | None
    content: MultimodalContent
    artifacts: frozenlist[DataModel]

    @property
    def should_respond(self) -> bool:
        return self.addressee is not None

    def response(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        artifacts: frozenlist[DataModel] = (),
        addressee: AgentID | None = None,
    ) -> "AgentMessage":
        assert not_missing(self.sender), "Missing message sender, can't respond to message drafts!"  # nosec: B101
        return AgentMessage(
            identifier=uuid4(),
            sender=MISSING,
            recipient=self.addressee or self.sender,
            addressee=addressee,
            content=MultimodalContent.of(content, *_content),
            artifacts=artifacts,
        )


type AgentInput = AsyncStream[AgentMessage]
type AgentOutput = frozenlist[AgentMessage] | AgentMessage | None


@runtime_checkable
class AgentStatePreparation[AgentState](Protocol):
    def __call__(self) -> AgentState: ...


@runtime_checkable
class AgentMemoryPreparation[AgentState, AgentStateScratch](Protocol):
    def __call__(self) -> Memory[AgentState, AgentStateScratch]: ...


@runtime_checkable
class AgentInvocation[AgentState, AgentStateScratch](Protocol):
    async def __call__(
        self,
        memory: Memory[AgentState, AgentStateScratch],
        message: AgentMessage,
    ) -> AgentOutput: ...


@runtime_checkable
class StatelessAgentInvocation(Protocol):
    async def __call__(
        self,
        message: AgentMessage,
    ) -> AgentOutput: ...


@final
class Agent[AgentState, AgentStateScratch]:
    def __init__(
        self,
        agent_id: AgentID,
        invocation: AgentInvocation[
            AgentState,
            AgentStateScratch,
        ],
        prepare_memory: AgentMemoryPreparation[AgentState, AgentStateScratch],
        concurrent: bool,
    ) -> None:
        self._agent_id: AgentID = agent_id
        self._agent_id._associate(agent=self)  # pyright: ignore[reportPrivateUsage]
        self._invocation: AgentInvocation[
            AgentState,
            AgentStateScratch,
        ] = invocation
        self._prepare_memory: AgentMemoryPreparation[AgentState, AgentStateScratch] = prepare_memory
        self._concurrent: bool = concurrent

    @property
    def identifier(self) -> UUID:
        return self._agent_id.identifier

    @property
    def name(self) -> str:
        return self._agent_id.name

    @property
    def capabilities(self) -> str | None:
        return self._agent_id.capabilities

    def address(
        self,
        content: MultimodalContent | MultimodalContentConvertible,
        /,
        *_content: MultimodalContent | MultimodalContentConvertible,
        artifacts: frozenlist[DataModel] = (),
        addressee: AgentID | None = None,
    ) -> AgentMessage:
        return self._agent_id.address(
            content,
            *_content,
            artifacts=artifacts,
            addressee=addressee,
        )

    async def process_message(
        self,
        message: AgentMessage,
        /,
        memory: Memory[AgentState, AgentStateScratch],
    ) -> AgentOutput:
        return await self._invocation(
            memory=memory,
            message=message,
        )

    def prepare_memory(self) -> Memory[AgentState, AgentStateScratch]:
        return self._prepare_memory()


class StatelessAgentWrapper(Protocol):
    def __call__(
        self,
        invocation: StatelessAgentInvocation,
    ) -> Agent[Stateless, Stateless]: ...


class PartialAgentWrapper[AgentState, AgentStateScratch](Protocol):
    def __call__(
        self,
        invocation: AgentInvocation[AgentState, AgentStateScratch],
    ) -> Agent[AgentState, AgentStateScratch]: ...


@overload
def agent(
    agent_id: AgentID,
    /,
    *,
    concurrent: bool = False,
) -> StatelessAgentWrapper: ...


@overload
def agent[AgentState, AgentStateScratch](
    agent_id: AgentID,
    /,
    *,
    state: AgentStatePreparation[AgentState],
) -> PartialAgentWrapper[AgentState, AgentState]: ...


@overload
def agent[AgentState, AgentStateScratch](
    agent_id: AgentID,
    /,
    *,
    memory: AgentMemoryPreparation[AgentState, AgentStateScratch],
) -> PartialAgentWrapper[AgentState, AgentStateScratch]: ...


@overload
def agent(
    *,
    name: str | None = None,
    capabilities: str,
    concurrent: bool = False,
) -> StatelessAgentWrapper: ...


@overload
def agent[AgentState, AgentStateScratch](
    *,
    name: str | None = None,
    capabilities: str,
    state: AgentStatePreparation[AgentState],
) -> PartialAgentWrapper[AgentState, AgentState]: ...


@overload
def agent[AgentState, AgentStateScratch](
    *,
    name: str | None = None,
    capabilities: str,
    memory: AgentMemoryPreparation[AgentState, AgentStateScratch],
) -> PartialAgentWrapper[AgentState, AgentStateScratch]: ...


def agent[AgentState, AgentStateScratch](  # noqa: PLR0913
    agent_id: AgentID | None = None,
    *,
    name: str | None = None,
    capabilities: str | None = None,
    state: AgentStatePreparation[AgentState] | None = None,
    memory: AgentMemoryPreparation[AgentState, AgentStateScratch] | None = None,
    concurrent: bool = False,
) -> StatelessAgentWrapper | PartialAgentWrapper[AgentState, AgentStateScratch]:
    assert state is None or memory is None, "Can't specify both state and memory"  # nosec: B101
    assert agent_id is None or (  # nosec: B101
        name is None and capabilities is None
    ), "Can't specify both agent id and name/capabilities"
    assert (  # nosec: B101
        capabilities is not None or agent_id is not None
    ), "Either agent id or capabilities has to be provided"

    def prepare_memory() -> Memory[AgentState, AgentStateScratch]:
        if memory is not None:
            return memory()

        elif state is not None:
            return cast(Memory[AgentState, AgentStateScratch], VolatileMemory(state()))

        else:
            return cast(Memory[AgentState, AgentStateScratch], ConstantMemory(Stateless()))

    def wrap(
        invocation: StatelessAgentInvocation | AgentInvocation[AgentState, AgentStateScratch],
    ) -> Agent[AgentState, AgentStateScratch]:
        assert isfunction(invocation), "Agent has to be defined from a function"  # nosec: B101

        return Agent[AgentState, AgentStateScratch](
            agent_id=agent_id
            or AgentID(
                name=name or invocation.__qualname__,
                capabilities=capabilities or "",
            ),
            prepare_memory=prepare_memory,
            invocation=invocation,
            concurrent=concurrent,
        )

    return wrap
