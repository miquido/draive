from collections.abc import AsyncIterable, Iterable, MutableSequence, Sequence
from typing import Any, NoReturn, Self, final, overload
from uuid import UUID, uuid4

from haiway import Meta, MetaValues, ctx

from draive.agents.state import AgentMemory
from draive.agents.types import (
    AgentExecuting,
    AgentIdentity,
    AgentMessage,
    AgentThread,
)
from draive.models import (
    GenerativeModel,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputSelection,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.models.types import ModelInput, ModelToolHandling
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
    MultimodalContentPart,
    Template,
    TemplatesRepository,
)
from draive.skills import Skill
from draive.steps import Step, StepState, StepStream
from draive.tools import Tool, Toolbox, tool
from draive.tools.types import ToolOutputChunk
from draive.utils import ProcessingEvent

__all__ = ("Agent",)


@final
class Agent:
    """Immutable async worker exposing a scoped streaming execution interface.

    Parameters
    ----------
    identity : AgentIdentity
        Immutable metadata identifying the agent.
    executing : AgentExecuting
        Async executor handling incoming messages.

    Attributes
    ----------
    identity : AgentIdentity
        Immutable metadata identifying the agent.
    """

    @overload
    @classmethod
    def noop(
        cls,
        agent: AgentIdentity,
    ) -> Self: ...

    @overload
    @classmethod
    def noop(
        cls,
        agent: str,
        *,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def noop(
        cls,
        agent: AgentIdentity | str,
        *,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a no operation agent.

        Parameters
        ----------
        agent : AgentIdentity | str
            Human-readable agent name or full identity.
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

        identity: AgentIdentity
        if isinstance(agent, str):
            identity = AgentIdentity.of(
                name=agent,
                description=description,
                meta=meta,
            )

        else:
            identity = agent

        return cls(
            identity=identity,
            executing=noop,
        )

    @overload
    @classmethod
    def generative(
        cls,
        agent: AgentIdentity,
        *,
        instructions: Template | ModelInstructions,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: AgentMemory = AgentMemory.disabled,
        output: ModelOutputSelection = "auto",
    ) -> Self: ...

    @overload
    @classmethod
    def generative(
        cls,
        agent: str,
        *,
        description: str = "",
        instructions: Template | ModelInstructions,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: AgentMemory = AgentMemory.disabled,
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def generative(  # noqa: C901, PLR0915
        cls,
        agent: AgentIdentity | str,
        *,
        description: str = "",
        instructions: Template | ModelInstructions,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: AgentMemory = AgentMemory.disabled,
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a model-backed agent using the completion-and-tools loop.

        Parameters
        ----------
        agent : AgentIdentity | str
            Human-readable agent name or full identity.
        description : str, default=""
            Short description of the agent's purpose.
        instructions : Template | ModelInstructions
            Instructions passed to the configured generative model.
        tools : Toolbox | Iterable[Tool], default=Toolbox.empty
            Tools available to the model while handling requests.
        memory : AgentMemory, default=AgentMemory.disabled
            Memory used to prepare and recall context before each turn and
            persist it afterwards. Defaults to a no-op memory scoped to a
            single turn. Context is remembered only when the turn completes
            and its output stream is fully consumed - turns abandoned
            mid-stream or failing with an error are not persisted. Tools
            returned by ``memory.prepare`` extend ``tools`` for the duration
            of the turn, replacing provided tools using the same names.
        output : ModelOutputSelection, default="auto"
            Output selection mode forwarded to model completion.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the agent identity.

        Returns
        -------
        Self
            Agent instance backed by ``Step.looping_completion(...)``.
        """
        identity: AgentIdentity
        if isinstance(agent, str):
            identity = AgentIdentity.of(
                name=agent,
                description=description,
                meta=meta,
            )

        else:
            identity = agent

        toolbox: Toolbox = Toolbox.of(tools)

        async def step(  # noqa: C901, PLR0912, PLR0915
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("agent.generative"):
                if isinstance(instructions, Template):
                    ctx.record_info(attributes={"instructions.template": instructions.identifier})
                    resolved_instructions: str = await TemplatesRepository.resolve_str(instructions)

                else:
                    resolved_instructions = instructions

                if not ctx.contains_state(AgentThread):
                    raise ValueError("AgentThread not specified")

                thread: AgentThread = ctx.state(AgentThread)

                memory_tools: Sequence[Tool] | None = await memory.prepare(
                    thread=thread,
                    instructions=resolved_instructions,
                )
                agent_tools: Toolbox
                if memory_tools:  # skip empty to avoid pointless toolbox copy
                    agent_tools = toolbox.with_tools(*memory_tools)

                else:
                    agent_tools = toolbox

                state = state.replacing_context(
                    await memory.recall(
                        thread=thread,
                        context=state.context,
                    )
                )

                iteration: int = 0
                while True:  # loop until we get ModelOutput without tools
                    async with ctx.scope(f"agent.generative.turn_{iteration}"):
                        ctx.log_debug("Generating completion...")
                        content_accumulator: MutableSequence[MultimodalContentPart] = []
                        reasoning_accumulator: MutableSequence[ModelReasoningChunk] = []
                        output_accumulator: MutableSequence[ModelOutputBlock] = []

                        async for chunk in GenerativeModel.completion(
                            instructions=resolved_instructions,
                            tools=agent_tools.model_tools(iteration=iteration),
                            context=state.context,
                            output=output,
                        ):
                            yield chunk

                            if isinstance(chunk, ModelReasoningChunk):
                                if content_accumulator:
                                    output_accumulator.append(
                                        MultimodalContent.of(*content_accumulator)
                                    )
                                    content_accumulator.clear()

                                reasoning_accumulator.append(chunk)

                            elif isinstance(chunk, ModelToolRequest):
                                # TODO: start handling immediately
                                if content_accumulator:
                                    output_accumulator.append(
                                        MultimodalContent.of(*content_accumulator)
                                    )
                                    content_accumulator.clear()

                                if reasoning_accumulator:
                                    output_accumulator.append(
                                        ModelReasoning.of(reasoning_accumulator)
                                    )
                                    reasoning_accumulator.clear()

                                output_accumulator.append(chunk)

                            else:
                                if reasoning_accumulator:
                                    output_accumulator.append(
                                        ModelReasoning.of(reasoning_accumulator)
                                    )
                                    reasoning_accumulator.clear()

                                content_accumulator.append(chunk)

                        if content_accumulator:
                            output_accumulator.append(MultimodalContent.of(*content_accumulator))

                        if reasoning_accumulator:
                            output_accumulator.append(ModelReasoning.of(reasoning_accumulator))

                        model_output: ModelOutput = ModelOutput.of(*output_accumulator)

                        state = state.appending_context(model_output)
                        yield state

                        tool_requests: Sequence[ModelToolRequest] = model_output.tool_requests
                        if not tool_requests:
                            break  # end of loop

                        ctx.log_debug("...handling tool requests...")

                        responses: MutableSequence[ModelToolResponse] = []
                        tools_output_accumulator: MutableSequence[MultimodalContentPart] = []
                        async for chunk in agent_tools.handle(*tool_requests):
                            if isinstance(chunk, ModelToolResponse):
                                responses.append(chunk)
                                yield chunk

                            elif isinstance(chunk, ProcessingEvent):
                                yield chunk

                            else:
                                tools_output_accumulator.append(chunk)
                                yield chunk

                        ctx.log_debug("...received tool responses...")

                        if tools_output_accumulator:  # tools direct result
                            ctx.log_debug("...tools generated output...")
                            state = state.appending_context(
                                ModelInput.of(*responses),
                                ModelOutput.of(MultimodalContent.of(*tools_output_accumulator)),
                            )
                            yield state
                            break  # end of loop

                        else:  # regular tools result
                            state = state.appending_context(
                                ModelInput.of(*responses),
                            )
                            yield state
                            iteration += 1  # continue next iteration

                await memory.remember(
                    thread=thread,
                    context=state.context,
                )

        return cls.steps(
            Step(step),
            agent=identity,
        )

    @overload
    @classmethod
    def from_skill(
        cls,
        skill: Skill,
        *,
        identity: AgentIdentity,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: AgentMemory = AgentMemory.disabled,
        output: ModelOutputSelection = "auto",
    ) -> Self: ...

    @overload
    @classmethod
    def from_skill(
        cls,
        skill: Skill,
        *,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: AgentMemory = AgentMemory.disabled,
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def from_skill(
        cls,
        skill: Skill,
        *,
        identity: AgentIdentity | None = None,
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        memory: AgentMemory = AgentMemory.disabled,
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a model-backed agent from a loaded Agent Skill.

        Parameters
        ----------
        skill : Skill
            Loaded skill containing discovery metadata, instructions, and
            optional resources.
        tools : Toolbox | Iterable[Tool], default=Toolbox.empty
            Additional tools available while handling requests.
        memory : AgentMemory, default=AgentMemory.disabled
            Memory used to prepare and recall context before each turn and
            persist it afterwards. Defaults to a no-op memory scoped to a
            single turn. Context is remembered only when the turn completes
            and its output stream is fully consumed - turns abandoned
            mid-stream or failing with an error are not persisted. Tools
            returned by ``memory.prepare`` extend ``tools`` for the duration
            of the turn, replacing provided tools using the same names.
        output : ModelOutputSelection, default="auto"
            Output selection mode forwarded to model completion.
        meta : Meta | MetaValues | None, default=None
            Additional metadata merged with the skill metadata and attached to
            the agent identity.

        Returns
        -------
        Self
            Agent instance configured from skill metadata and instructions.
        """

        if identity is None:
            identity = AgentIdentity.of(
                name=skill.name,
                description=skill.description,
                meta=skill.meta.merged_with(meta),
            )

        resolved_toolbox: Toolbox
        if isinstance(tools, Toolbox):
            resolved_toolbox = tools.with_tools(skill.resources_tool())

        else:
            resolved_toolbox = Toolbox.of(*tools, skill.resources_tool())

        return cls.generative(
            identity,
            instructions=skill.instructions,
            tools=resolved_toolbox,
            memory=memory,
            output=output,
        )

    @overload
    @classmethod
    def steps(
        cls,
        /,
        step: Step,
        *steps: Step,
        agent: AgentIdentity,
    ) -> Self: ...

    @overload
    @classmethod
    def steps(
        cls,
        /,
        step: Step,
        *steps: Step,
        agent: str,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def steps(
        cls,
        /,
        step: Step,
        *steps: Step,
        agent: AgentIdentity | str,
        description: str = "",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create an agent from one or more ``Step`` pipeline stages.

        Parameters
        ----------
        step : Step
            First step executed on the initial context holding the incoming
            message as input.
        *steps : Step
            Additional steps executed sequentially after ``step``.
        agent : AgentIdentity | str
            Human-readable agent name or full identity.
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
        The wrapped execution seeds the pipeline context with the incoming
        message as ``ModelInput``, runs ``step`` and ``*steps``, and filters
        out reasoning and tool protocol chunks from the public output stream.
        Memory is not applied implicitly - compose ``AgentMemory`` steps
        (``recall_step``, ``remember_step``) into the pipeline where needed.
        There is no step equivalent of ``memory.prepare`` - a step cannot
        contribute the tools it produces to the toolbox of a subsequent
        completion step - so preparation and memory tools are available only
        in ``generative`` and ``from_skill``, which invoke ``memory.prepare``,
        ``memory.recall``, and ``memory.remember`` directly within their step
        bodies.
        """

        async def execute(
            message: AgentMessage,
        ) -> AsyncIterable[MultimodalContentPart | ProcessingEvent]:
            async for chunk in Step.sequence(
                step,
                *steps,
            ).stream(
                (
                    ModelInput.of(
                        message.content,
                        meta=message.meta,
                    ),
                )
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

        identity: AgentIdentity
        if isinstance(agent, str):
            identity = AgentIdentity.of(
                name=agent,
                description=description,
                meta=meta,
            )

        else:
            identity = agent

        return cls(
            identity=identity,
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
            ``AgentThread`` thread is reused or a new one is created.
        input : Multimodal
            Input payload converted into an ``AgentMessage``.
        meta : Meta | MetaValues | None, default=None
            Metadata merged into the active ``AgentThread``.

        Returns
        -------
        AsyncIterable[MultimodalContentPart | ProcessingEvent]
            Stream of output chunks emitted by the agent.
        """
        current: AgentThread | None
        if ctx.contains_state(AgentThread):
            current = ctx.state(AgentThread)

        else:
            current = None

        identifier: UUID
        if thread is not None:
            identifier = thread

        elif current is not None:
            identifier = current.identifier

        else:
            identifier = uuid4()

        return self.respond(
            AgentMessage(
                thread=identifier,
                content=MultimodalContent.of(input),
                meta=current.meta.merged_with(meta) if current is not None else Meta.of(meta),
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
            AgentThread.of(
                message.thread,
                agent_uri=self.identity.uri,
                meta=message.meta,
            ),
        ):
            ctx.log_info(
                f"Agent {self.identity.name} responding to message within thread {message.thread}"
            )
            async for chunk in self._executing(message):
                yield chunk

    def as_tool(  # noqa: C901
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        handling: ModelToolHandling = "response",
        meta: Meta | MetaValues | None = None,
    ) -> Tool:
        """Expose the agent as a callable tool.

        Parameters
        ----------
        name : str | None, default=None
            Tool name. When omitted, a name is derived from the agent identity
            and requested handling mode.
        description : str | None, default=None
            Tool description. When omitted, a description is derived from the
            agent identity and requested handling mode.
        handling : ModelToolHandling, default="response"
            Tool handling mode used when registering the generated tool.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the generated tool definition.

        Returns
        -------
        Tool
            Tool forwarding its ``task`` input to the agent.
        """
        if name is None:
            match handling:
                case "response":
                    name = f"agent_{self.identity.name}_request"

                case "output" | "output_stream":
                    name = f"agent_{self.identity.name}_handover"

        if description is None:
            match handling:
                case "response":
                    description = (
                        f"Request the {self.identity.name} agent to perform a task for you.\n"
                        f"\n{self.identity.description}"
                    )

                case "output" | "output_stream":
                    description = (
                        f"Hand over your task to the {self.identity.name} agent.\n"
                        f"\n{self.identity.description}"
                    )

        task_description: str
        match handling:
            case "response":
                task_description = "Task to be performed by the agent"

            case "output" | "output_stream":
                task_description = "Task to be handed over to the agent"

        @tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": task_description,
                    },
                },
                "required": ("task",),
                "additionalProperties": False,
            },
            handling=handling,
            meta=meta,
        )
        async def agent_request(
            task: str,
        ) -> AsyncIterable[ToolOutputChunk]:
            async for chunk in self.call(input=task):
                yield chunk

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
