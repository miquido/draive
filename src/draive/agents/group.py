from collections.abc import (
    AsyncGenerator,
    Mapping,
    MutableMapping,
)
from typing import Any, NoReturn, Self, final
from uuid import UUID

from haiway import Map, Meta, MetaValues

from draive.agents.agent import Agent
from draive.agents.types import AgentException, AgentIdentity, AgentMessage, AgentUnavailable
from draive.models.types import ModelToolHandling
from draive.multimodal import Multimodal, MultimodalContentPart
from draive.tools import Tool, ToolOutputChunk, tool
from draive.utils import ProcessingEvent

__all__ = ("AgentsGroup",)


@final
class AgentsGroup:
    """Registry of agents indexed by stable URI, supporting replacement.

    The group provides direct lookup by agent name or URI and can expose the
    registered agents as tools for model-driven delegation.
    """

    @classmethod
    def of(
        cls,
        *agents: Agent | AgentIdentity,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent group indexed by agent URI.

        Parameters
        ----------
        *agents : Agent | AgentIdentity
            Agents to include in the group.
        meta : Meta | MetaValues | None, default=None
            Metadata attached to the group.

        Returns
        -------
        Self
            Immutable agent group instance.

        Raises
        ------
        ValueError
            Raised when duplicate or ambiguous agent names or URIs are provided.
        """

        available: MutableMapping[str, Agent] = {}
        for agent_or_identity in agents:
            agent: Agent
            if isinstance(agent_or_identity, AgentIdentity):
                agent = Agent(
                    identity=agent_or_identity,
                    executing=_undefined_agent,
                )

            else:
                assert isinstance(agent_or_identity, Agent)  # nosec: B101
                agent = agent_or_identity

            if agent.identity.uri in available:
                raise ValueError(f"Agent `{agent.identity.uri}` is already defined")

            available[agent.identity.uri] = agent

        return cls(
            agents=available,
            meta=Meta.of(meta),
        )

    __slots__ = (
        "_agents",
        "_names",
        "meta",
    )

    def __init__(
        self,
        agents: Mapping[str, Agent],
        meta: Meta = Meta.empty,
    ) -> None:
        """Initialize an agent group from agents indexed by URI.

        Parameters
        ----------
        agents : Mapping[str, Agent]
            Concrete or placeholder agents indexed by their identity URI.
        meta : Meta, default=Meta.empty
            Metadata attached to the group itself.

        Raises
        ------
        ValueError
            Raised when keys differ from identity URIs or names are ambiguous.
        """
        self._agents: Mapping[str, Agent]
        object.__setattr__(
            self,
            "_agents",
            Map(agents),  # make a copy
        )
        self._names: Mapping[str, str]
        object.__setattr__(
            self,
            "_names",
            Map({agent.identity.name: agent.identity.uri for agent in agents.values()}),
        )
        assert len(self._agents) == len(self._names)  # nosec: B101
        self.meta: Meta
        object.__setattr__(
            self,
            "meta",
            meta,
        )

    def bind(
        self,
        agent: Agent,
    ) -> None:
        """Bind a concrete agent to an existing entry.

        Parameters
        ----------
        agent : Agent
            Agent instance whose URI must match an existing entry. Both
            placeholders and concrete agents can be replaced.

        Returns
        -------
        None
            This method replaces the registered agent in place.

        Raises
        ------
        AgentException
            Raised when the URI was not declared or the name conflicts with
            another registered agent's name or URI.
        """
        if agent.identity.uri not in self._agents:
            raise AgentException("AgentGroup agents can't be extended")

        matching = self._resolve(agent.identity.name)
        if matching is not None and matching.identity.uri != agent.identity.uri:
            raise AgentException(f"Agent `{agent.identity.name}` is already defined")

        names = dict(self._names)
        del names[self._agents[agent.identity.uri].identity.name]
        names[agent.identity.name] = agent.identity.uri

        object.__setattr__(
            self,
            "_agents",
            Map(
                {
                    **self._agents,
                    agent.identity.uri: agent,
                }
            ),
        )
        object.__setattr__(
            self,
            "_names",
            Map(names),
        )
        assert len(self._agents) == len(self._names)  # nosec: B101

    def _resolve(
        self,
        reference: str,
    ) -> Agent | None:
        if selected := self._agents.get(reference):
            return selected

        if uri := self._names.get(reference):
            return self._agents[uri]

        return None

    async def call(
        self,
        agent: str,
        *,
        thread: UUID | None = None,
        input: Multimodal,  # noqa: A002
        meta: Meta | MetaValues | None = None,
    ) -> AsyncGenerator[MultimodalContentPart | ProcessingEvent]:
        """Call a selected agent directly through the group.

        Parameters
        ----------
        agent : str
            URI or name of the agent to execute.
        thread : UUID | None, default=None
            Conversation thread identifier forwarded to the selected agent.
        input : Multimodal
            Input payload forwarded to the selected agent.
        meta : Meta | MetaValues | None, default=None
            Metadata forwarded to the selected agent call.

        Returns
        -------
        AsyncGenerator[MultimodalContentPart | ProcessingEvent]
            Stream of chunks emitted by the selected agent.

        Raises
        ------
        AgentUnavailable
            Raised when the referenced agent name is not defined in the group.
        """
        if selected := self._resolve(agent):
            agent_stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = selected.call(
                thread=thread,
                input=input,
                meta=meta,
            )
            try:
                async for chunk in agent_stream:
                    yield chunk

            finally:
                # release the agent when the consumer stops before its end
                await agent_stream.aclose()

        else:
            raise AgentUnavailable(f"Agent `{agent}` is not defined")

    def as_tool(  # noqa: C901
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        handling: ModelToolHandling = "response",
        meta: Meta | MetaValues | None = None,
    ) -> Tool:
        """Expose the declared agents as a model-callable tool.

        Parameters
        ----------
        name : str | None, default=None
            Explicit tool name. When omitted, a name is derived from
            ``handling``.
        description : str | None, default=None
            Explicit tool description. When omitted, a description listing the
            declared agents is generated automatically.
        handling : ModelToolHandling, default="response"
            Tool handling mode used to determine both the generated defaults and
            how the resulting tool is interpreted by the model runtime.
        meta : Meta | MetaValues | None, default=None
            Metadata attached to the generated tool.

        Returns
        -------
        Tool
            Tool that accepts an agent name and task, then delegates execution
            to the selected agent.

        Raises
        ------
        AgentUnavailable
            Raised when the generated tool is invoked with an agent name that is
            declared in the schema but not currently bound.
        """
        if name is None:
            match handling:
                case "response":
                    name = "agent_request"

                case "output":
                    name = "agent_handover"

        if description is None:
            match handling:
                case "response":
                    description = "Request one of available agents to perform a task for you:\n"
                    description += "\n".join(
                        f'<agent name="{agent.identity.name}">{agent.identity.description}</agent>'
                        for agent in self._agents.values()
                    )

                case "output":
                    description = "Hand over your task to one of available agents:\n"
                    description += "\n".join(
                        f'<agent name="{agent.identity.name}">{agent.identity.description}</agent>'
                        for agent in self._agents.values()
                    )

        task_description: str
        match handling:
            case "response":
                task_description = "Task to be performed by the selected agent"

            case "output":
                task_description = "Task to be handed over to the selected agent"

        @tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": tuple(agent.identity.name for agent in self._agents.values()),
                        "description": "Selected agent name",
                    },
                    "task": {
                        "type": "string",
                        "description": task_description,
                    },
                },
                "required": (
                    "agent",
                    "task",
                ),
                "additionalProperties": False,
            },
            handling=handling,
            meta=meta,
        )
        async def agent_request(
            agent: str,
            task: str,
        ) -> AsyncGenerator[ToolOutputChunk]:
            if selected := self._resolve(agent):
                agent_stream: AsyncGenerator[MultimodalContentPart | ProcessingEvent] = (
                    selected.call(input=task)
                )
                try:
                    async for chunk in agent_stream:
                        yield chunk

                finally:
                    await agent_stream.aclose()

            else:
                raise AgentUnavailable(f"Agent `{agent}` is not defined")

        return agent_request

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> NoReturn:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be modified"
        )

    def __delattr__(
        self,
        name: str,
    ) -> NoReturn:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be deleted"
        )


async def _undefined_agent(
    message: AgentMessage,
) -> AsyncGenerator[MultimodalContentPart | ProcessingEvent]:
    raise AgentUnavailable("Agent execution method undefined!")
    yield  # converts to AsyncGenerator
