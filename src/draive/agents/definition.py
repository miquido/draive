from inspect import isfunction
from typing import Protocol, cast, overload, runtime_checkable

from draive.agents.node import Agent, AgentMessage, AgentNode, AgentOutput
from draive.helpers import VolatileMemory
from draive.types import Memory

__all__ = [
    "agent",
    "AgentStateInitializer",
    "AgentMemoryInitializer",
    "AgentInvocation",
]


@runtime_checkable
class AgentStateInitializer[AgentState](Protocol):
    def __call__(self) -> AgentState: ...


@runtime_checkable
class AgentMemoryInitializer[AgentState, AgentStateScratch](Protocol):
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


class PartialAgentWrapper[AgentState, AgentStateScratch](Protocol):
    def __call__(
        self,
        invocation: AgentInvocation[AgentState, AgentStateScratch],
    ) -> AgentNode: ...


class PartialStatelessAgentWrapper(Protocol):
    def __call__(
        self,
        invocation: StatelessAgentInvocation,
    ) -> AgentNode: ...


@overload
def agent(
    node: AgentNode,
    /,
) -> PartialStatelessAgentWrapper: ...


@overload
def agent[AgentState](
    node: AgentNode,
    /,
    *,
    state: AgentStateInitializer[AgentState],
) -> PartialAgentWrapper[AgentState, AgentState]: ...


@overload
def agent[AgentState, AgentStateScratch](
    node: AgentNode,
    /,
    *,
    memory: AgentMemoryInitializer[AgentState, AgentStateScratch],
) -> PartialAgentWrapper[AgentState, AgentStateScratch]: ...


@overload
def agent(
    *,
    name: str | None = None,
    description: str,
) -> PartialStatelessAgentWrapper: ...


@overload
def agent[AgentState](
    *,
    name: str | None = None,
    description: str,
    state: AgentStateInitializer[AgentState],
) -> PartialAgentWrapper[AgentState, AgentState]: ...


@overload
def agent[AgentState, AgentStateScratch](
    *,
    name: str | None = None,
    description: str,
    memory: AgentMemoryInitializer[AgentState, AgentStateScratch],
) -> PartialAgentWrapper[AgentState, AgentStateScratch]: ...


def agent[AgentState, AgentStateScratch](
    node: AgentNode | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    state: AgentStateInitializer[AgentState] | None = None,
    memory: AgentMemoryInitializer[AgentState, AgentStateScratch] | None = None,
) -> PartialStatelessAgentWrapper | PartialAgentWrapper[AgentState, AgentStateScratch]:
    assert state is None or memory is None, "Can't specify both state and memory"  # nosec: B101
    assert node is None or (  # nosec: B101
        name is None and description is None
    ), "Can't specify both agent node and name/description"
    assert (  # nosec: B101
        description is not None or node is not None
    ), "Either agent node or description has to be provided"

    def wrap(
        invocation: AgentInvocation[AgentState, AgentStateScratch],
    ) -> AgentNode:
        assert isfunction(invocation), "agent has to be defined from a function"  # nosec: B101

        agent_node: AgentNode = node or AgentNode(
            name=name or invocation.__qualname__,
            description=description or "",
        )
        concurrent: bool

        if memory is not None:
            concurrent = False
            memory_initializer: AgentMemoryInitializer[AgentState, AgentStateScratch] = memory

            def initialize() -> Agent:
                agent_memory: Memory[AgentState, AgentStateScratch] = memory_initializer()

                async def agent(message: AgentMessage) -> AgentOutput:
                    return await invocation(
                        memory=agent_memory,
                        message=message,
                    )

                return agent

        elif state is not None:
            concurrent = False
            state_initializer: AgentStateInitializer[AgentState] = state

            def initialize() -> Agent:
                agent_memory: Memory[AgentState, AgentStateScratch] = cast(
                    Memory[AgentState, AgentStateScratch], VolatileMemory(state_initializer())
                )

                async def agent(message: AgentMessage) -> AgentOutput:
                    return await invocation(
                        memory=agent_memory,
                        message=message,
                    )

                return agent

        else:
            concurrent = True  # stateless agents are concurrent by default

            def initialize() -> Agent:
                async def stateless_agent(message: AgentMessage) -> AgentOutput:
                    return await invocation(message=message)

                return stateless_agent

        agent_node._associate(  # pyright: ignore[reportPrivateUsage]
            initialize,
            concurrent=concurrent,
        )

        return agent_node

    return wrap
