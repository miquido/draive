from collections.abc import Callable
from inspect import isfunction
from typing import final, overload
from uuid import uuid4

from draive.agents.abc import BaseAgent
from draive.agents.errors import AgentException
from draive.agents.state import AgentState
from draive.agents.types import AgentInvocation
from draive.metrics import ArgumentsTrace
from draive.parameters import ParametrizedData
from draive.scope import ctx
from draive.types import MultimodalContent
from draive.utils import freeze

__all__ = [
    "agent",
    "Agent",
]


@final
class Agent[State: ParametrizedData](BaseAgent[State]):
    def __init__(
        self,
        name: str,
        description: str,
        invoke: AgentInvocation[State],
    ) -> None:
        self.invoke: AgentInvocation[State] = invoke
        super().__init__(
            agent_id=uuid4().hex,
            name=name,
            description=description,
        )

        freeze(self)

    async def __call__(
        self,
        state: AgentState[State],
    ) -> MultimodalContent | None:
        invocation_id: str = uuid4().hex
        with ctx.nested(
            f"Agent|{self.name}",
            metrics=[
                ArgumentsTrace.of(
                    agent_id=self.agent_id,
                    invocation_id=invocation_id,
                    state=state,
                )
            ],
        ):
            try:
                return await self.invoke(state)

            except Exception as exc:
                raise AgentException(
                    "Agent invocation %s of %s failed due to an error: %s",
                    invocation_id,
                    self.agent_id,
                    exc,
                ) from exc


@overload
def agent[State: ParametrizedData](
    invoke: AgentInvocation[State],
    /,
) -> Agent[State]: ...


@overload
def agent[State: ParametrizedData](
    *,
    name: str,
    description: str | None = None,
) -> Callable[[AgentInvocation[State]], Agent[State]]: ...


@overload
def agent[State: ParametrizedData](
    *,
    description: str,
) -> Callable[[AgentInvocation[State]], Agent[State]]: ...


def agent[State: ParametrizedData](
    invoke: AgentInvocation[State] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[AgentInvocation[State]], Agent[State]] | Agent[State]:
    def wrap(
        invoke: AgentInvocation[State],
    ) -> Agent[State]:
        assert isfunction(invoke), "Agent has to be defined from function"  # nosec: B101
        return Agent[State](
            name=name or invoke.__qualname__,
            description=description or "",
            invoke=invoke,
        )

    if invoke := invoke:
        return wrap(invoke=invoke)
    else:
        return wrap
