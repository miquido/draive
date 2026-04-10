from collections.abc import AsyncIterable, Collection, Mapping, MutableMapping, MutableSet, Set
from typing import Any, NoReturn, Self, final
from uuid import UUID

from haiway import Meta, MetaValues

from draive.agents.agent import Agent
from draive.agents.types import AgentException, AgentIdentity, AgentUnavailable
from draive.models.types import ModelToolHandling
from draive.multimodal import Multimodal, MultimodalContentPart
from draive.tools import Tool, ToolOutputChunk, tool
from draive.utils import ProcessingEvent

__all__ = ("AgentsGroup",)


@final
class AgentsGroup:
    """Immutable registry of declared and bound agents.

    The group provides direct lookup by agent name or URI and can expose the
    registered agents as tools for model-driven delegation.
    """

    @classmethod
    def of(
        cls,
        *agents: Agent | AgentIdentity,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent group indexed by agent name.

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
            Raised when duplicate agent names are provided.
        """

        declared: MutableSet[AgentIdentity] = set()
        available: MutableMapping[str, Agent] = {}
        for agent in agents:
            if isinstance(agent, AgentIdentity):
                if agent in declared:
                    raise ValueError(f"Agent `{agent}` is already defined")

                if agent.uri in available:
                    raise ValueError(f"Agent `{agent.uri}` is already defined")

                if agent.name in available:
                    raise ValueError(f"Agent `{agent.name}` is already defined")

                declared.add(agent)

            else:
                if agent.identity in declared:
                    raise ValueError(f"Agent `{agent}` is already defined")

                if agent.identity.uri in available:
                    raise ValueError(f"Agent `{agent.identity.uri}` is already defined")

                if agent.identity.name in available:
                    raise ValueError(f"Agent `{agent.identity.name}` is already defined")

                declared.add(agent.identity)
                # associate both uri and name with the agent
                available[agent.identity.uri] = agent
                available[agent.identity.name] = agent

        return cls(
            declared=declared,
            available=available,
            meta=Meta.of(meta),
        )

    __slots__ = (
        "_available",
        "_declared",
        "meta",
    )

    def __init__(
        self,
        available: Mapping[str, Agent],
        declared: Set[AgentIdentity] | None = None,
        meta: Meta = Meta.empty,
    ) -> None:
        """Initialize an agent group from explicit declarations and bindings.

        Parameters
        ----------
        available : Mapping[str, Agent]
            Concrete agents indexed by agent URI and/or name.
        declared : Set[AgentIdentity] | None, default=None
            Full set of identities allowed in this group. When omitted, the
            identities are inferred from ``available``.
        meta : Meta, default=Meta.empty
            Metadata attached to the group itself.

        Raises
        ------
        AssertionError
            Raised in debug mode when any bound agent identity is missing from
            the declared set.
        """
        self._declared: Collection[AgentIdentity]
        if declared is None:
            object.__setattr__(
                self,
                "_declared",
                frozenset(element.identity for element in available.values()),
            )

        else:
            object.__setattr__(
                self,
                "_declared",
                frozenset(declared),
            )

        assert all(element.identity in self._declared for element in available.values())  # nosec: B101

        self._available: Mapping[str, Agent]
        object.__setattr__(
            self,
            "_available",
            available,
        )
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
        """Bind a concrete agent to an existing placeholder entry.

        Parameters
        ----------
        agent : Agent
            Agent instance whose name must match a placeholder defined when the
            group was created.

        Returns
        -------
        None
            This method updates the internal placeholder mapping in place.

        Raises
        ------
        AgentException
            Raised when the agent name was not predeclared in the group or when
            a concrete agent is already bound under that name.
        """
        if agent.identity not in self._declared:
            raise AgentException("AgentGroup agents can't be extended")

        matching: Agent | None = self._available.get(
            agent.identity.uri,
            self._available.get(
                agent.identity.name,
            ),
        )

        if matching is not None:
            raise AgentException("AgentGroup agents can't be redefined")

        object.__setattr__(
            self,
            "_available",
            {
                **self._available,
                # associate both uri and name with the agent
                agent.identity.uri: agent,
                agent.identity.name: agent,
            },
        )

    async def call(
        self,
        agent: str,
        *,
        thread: UUID | None = None,
        input: Multimodal,  # noqa: A002
        meta: Meta | MetaValues | None = None,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
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
        AsyncIterable[MultimodalContentPart | ProcessingEvent]
            Stream of chunks emitted by the selected agent.

        Raises
        ------
        AgentUnavailable
            Raised when the referenced agent name is not defined in the group.
        """
        if selected := self._available.get(agent):
            async for chunk in selected.call(
                thread=thread,
                input=input,
                meta=meta,
            ):
                yield chunk

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
                    description = "Request the selected agent to perform a task for you.\n"
                    description += "\n".join(
                        f'<agent name="{identity.name}">{identity.description}</agent>'
                        for identity in self._declared
                    )

                case "output":
                    description = "Hand over your task to the selected agent.\n"
                    description += "\n".join(
                        f'<agent name="{identity.name}">{identity.description}</agent>'
                        for identity in self._declared
                    )

        task_description: str
        match handling:
            case "response":
                task_description = "Task to be performed by the agent"

            case "output":
                task_description = "Task to be handed over to the agent"

        @tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": tuple(identity.name for identity in self._declared),
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
        ) -> AsyncIterable[ToolOutputChunk]:
            if selected := self._available.get(agent):
                async for chunk in selected.call(input=task):
                    yield chunk

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
