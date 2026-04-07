from collections.abc import AsyncIterable, Iterable
from typing import Any, NoReturn, Self, final
from uuid import UUID, uuid4

from haiway import Meta, MetaValues, ctx

from draive.agents.types import (
    AgentContext,
    AgentExecuting,
    AgentIdentity,
    AgentMessage,
)
from draive.models import (
    ModelInstructions,
    ModelOutputSelection,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
    MultimodalContentPart,
    Template,
)
from draive.steps import Step
from draive.tools import Tool, Toolbox
from draive.utils import ProcessingEvent

__all__ = ("Agent",)


@final
class Agent:
    """Immutable async worker exposing a scoped streaming execution interface."""

    @classmethod
    def noop(
        cls,
        name: str,
        *,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a no operation agent.

        Parameters
        ----------
        name : str
            Human-readable agent name.
        description : str, default=""
            Short description of the agent's purpose.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the agent identity.

        Returns
        -------
        Self
            Agent without operations.
        """

        async def noop(
            message: AgentMessage,
        ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
            return  # do not emit anything
            yield  # converts to AsyncGenerator

        return cls(
            identity=AgentIdentity(
                uri=f"agent://{uuid4()}",
                name=name,
                description=description,
                meta=Meta.of(meta),
            ),
            executing=noop,
        )

    @classmethod
    def generative(
        cls,
        name: str,
        *,
        description: str = "",
        instructions: Template | ModelInstructions,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a model-backed agent using the completion-and-tools loop.

        Parameters
        ----------
        name : str
            Human-readable agent name.
        description : str, default=""
            Short description of the agent's purpose.
        instructions : Template | ModelInstructions
            Instructions passed to the configured generative model.
        tools : Toolbox | Iterable[Tool], default=Toolbox.empty
            Tools available to the model while handling requests.
        output : ModelOutputSelection, default="auto"
            Output selection mode forwarded to model completion.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the agent identity.

        Returns
        -------
        Self
            Agent instance backed by ``Step.looping_completion(...)``.
        """
        return cls.steps(
            Step.looping_completion(
                instructions=instructions,
                tools=tools,
                output=output,
            ),
            name=name,
            description=description,
            meta=meta,
        )

    @classmethod
    def steps(
        cls,
        /,
        step: Step,
        *steps: Step,
        name: str,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent from one or more ``Step`` pipeline stages.

        Parameters
        ----------
        step : Step
            First step executed after the incoming message is appended as input.
        *steps : Step
            Additional steps executed sequentially after ``step``.
        name : str
            Human-readable agent name.
        description : str, default=""
            Short description of the agent's purpose.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the agent identity.

        Returns
        -------
        Self
            Agent instance exposing only visible content and processing events.

        Notes
        -----
        The wrapped execution prepends ``Step.appending_input(message.content)``
        and filters out reasoning and tool protocol chunks from the public
        output stream.
        """

        async def execute(
            message: AgentMessage,
        ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
            async for chunk in Step.sequence(
                Step.appending_input(message.content),
                step,
                *steps,
            ):
                if isinstance(chunk, ModelReasoningChunk):
                    continue  # skip reasoning

                elif isinstance(chunk, ProcessingEvent):
                    yield chunk  # pass events

                elif isinstance(chunk, ModelToolRequest):
                    continue  # skip tools within output

                elif isinstance(chunk, ModelToolResponse):
                    continue  # skip tools within output

                else:
                    yield chunk  # pass content

        return cls(
            identity=AgentIdentity(
                uri=f"agent://{uuid4()}",
                name=name,
                description=description,
                meta=Meta.of(meta),
            ),
            executing=execute,
        )

    __slots__ = (
        "_executing",
        "identity",
    )

    identity: AgentIdentity
    _executing: AgentExecuting

    def __init__(
        self,
        identity: AgentIdentity,
        executing: AgentExecuting,
    ) -> None:
        """Initialize an agent from explicit identity and executor.

        Parameters
        ----------
        identity : AgentIdentity
            Immutable metadata identifying the agent.
        executing : AgentExecuting
            Async executor handling incoming messages.
        """
        self.identity: AgentIdentity
        object.__setattr__(
            self,
            "identity",
            identity,
        )
        self._executing: AgentExecuting
        object.__setattr__(
            self,
            "_executing",
            executing,
        )

    def call(
        self,
        *,
        thread: UUID | None = None,
        input: Multimodal,  # noqa: A002
        meta: Meta | MetaValues | None = None,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        """Call the agent with raw multimodal input.

        Parameters
        ----------
        thread : UUID | None, default=None
            Conversation thread identifier. When omitted, the current
            ``AgentContext`` thread is reused or a new one is created.
        input : Multimodal
            Input payload converted into an ``AgentMessage``.
        meta : Meta | MetaValues | None, default=None
            Metadata merged into the active ``AgentContext``.

        Returns
        -------
        AsyncIterable[MultimodalContentPart | ProcessingEvent]
            Stream of output chunks emitted by the agent.
        """
        context: AgentContext = ctx.state(
            AgentContext,
            default=AgentContext.of(),
        )
        return self.respond(
            AgentMessage(
                thread=thread if thread is not None else context.thread,
                content=MultimodalContent.of(input),
                meta=context.meta.merged_with(meta),
            ),
        )

    async def respond(
        self,
        message: AgentMessage,
    ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
        """Execute the agent for an already prepared message.

        Parameters
        ----------
        message : AgentMessage
            Message to process.

        Returns
        -------
        AsyncIterable[MultimodalContentPart | ProcessingEvent]
            Stream of output chunks emitted while handling the message.
        """
        async with ctx.scope(
            f"agent.{self.identity.name}",
            AgentContext.of(
                thread=message.thread,
                meta=message.meta,
            ),
        ):
            async for chunk in self._executing(message):
                yield chunk

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
