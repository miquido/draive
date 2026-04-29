from asyncio import ALL_COMPLETED, Task, sleep, wait
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
)
from inspect import iscoroutinefunction
from typing import Any, ClassVar, NoReturn, Protocol, Self, final, overload, runtime_checkable

from haiway import (
    AsyncStream,
    Disposable,
    Disposables,
    Meta,
    MetaValues,
    State,
    ctx,
)
from haiway.context.tasks import ContextTaskGroup

from draive.evaluation import (
    EvaluatorResult,
    EvaluatorScenarioResult,
    PreparedEvaluator,
    PreparedEvaluatorScenario,
)
from draive.models import (
    GenerativeModel,
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputSelection,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    ModelToolSpecification,
)
from draive.multimodal import (
    ArtifactContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentPart,
    Template,
    TemplatesRepository,
)
from draive.steps.state import StepState
from draive.steps.types import (
    StepConditionVerifying,
    StepContextMutating,
    StepException,
    StepExecuting,
    StepLoopConditionVerifying,
    StepMerging,
    StepOutputChunk,
    StepProcessing,
    StepStatePreserving,
    StepStateRestoring,
    StepStream,
)
from draive.tools import Tool, Toolbox
from draive.utils import ProcessingEvent

__all__ = (
    "Step",
    "step",
)


@final  # consider immutable
class Step:
    """Composable asynchronous processing unit operating on ``StepState``.

    ``Step`` wraps an async callable that transforms a ``StepState`` and may
    emit streamed output chunks while executing. Class and instance helpers
    provide higher-level patterns such as prompting, generation, tool handling,
    sequencing, looping, branching, retries, fallbacks, and evaluation gates.

    Notes
    -----
    Rationale: this abstraction keeps step orchestration strongly typed and
    context-aware while staying easy to compose.
    """

    noop: ClassVar[Self]  # defined after the class

    @classmethod
    def emitting(
        cls,
        *parts: StepOutputChunk,
    ) -> Self:
        """Create a step that emits fixed output parts and leaves state unchanged.

        Parameters
        ----------
        *parts : StepOutputChunk
            Output parts to emit in order.

        Returns
        -------
        Self
            A step emitting the provided parts, or ``Step.noop`` when no parts
            are provided.

        Notes
        -----
        Rationale: useful for injecting deterministic stream output into a
        processing pipeline.
        """
        if not parts:
            return cls.noop

        async def step(
            state: StepState,
        ) -> StepStream:
            for part in parts:
                yield part

        return cls(step)

    @classmethod
    def updating_artifacts(
        cls,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> Self:
        """Create a step that updates artifacts in ``StepState``.

        Parameters
        ----------
        *artifacts : State
            Artifact states stored under their class names.
        **keyed_artifacts : State
            Artifact states stored under explicit keys.

        Returns
        -------
        Self
            A step merging provided artifacts into state, or ``Step.noop`` when
            no artifacts are provided.

        Notes
        -----
        Rationale: enables declarative artifact injection without custom step
        implementations.
        """
        if not artifacts and not keyed_artifacts:
            return cls.noop

        merged_artifacts: Mapping[str, ArtifactContent] = {
            **{item.__class__.__name__: ArtifactContent.of(item) for item in artifacts},
            **{key: ArtifactContent.of(value) for key, value in keyed_artifacts.items()},
        }

        async def step(
            state: StepState,
        ) -> StepStream:
            yield state.updating(
                artifacts={
                    **state.artifacts,
                    **merged_artifacts,
                }
            )

        return cls(step)

    @classmethod
    def appending_context(
        cls,
        *elements: ModelContextElement,
    ) -> Self:
        """Create a step appending model-context elements.

        Parameters
        ----------
        *elements : ModelContextElement
            Context elements appended to existing context in order.

        Returns
        -------
        Self
            A step extending context, or ``Step.noop`` when no elements are
            provided.

        Notes
        -----
        Rationale: adds reusable context mutations while preserving immutability
        of previous state values.
        """
        if not elements:
            return cls.noop

        async def step(
            state: StepState,
        ) -> StepStream:
            yield state.updating(
                context=(
                    *state.context,
                    *elements,
                )
            )

        return cls(step)

    @classmethod
    def replacing_context(
        cls,
        context: Callable[[], Coroutine[None, None, ModelContext]] | ModelContext,
    ) -> Self:
        """Create a step assigning fixed context.

        Parameters
        ----------
        context : Callable[[], Coroutine[None, None, ModelContext]] | ModelContext
            Replacement model context or async provider returning it.

        Returns
        -------
        Self
            A step that replaces context with the provided value.

        Notes
        -----
        Rationale: explicit context rewrite logic.
        """

        if iscoroutinefunction(context):

            async def step(
                state: StepState,
            ) -> StepStream:
                yield state.updating(context=await context())

        else:

            async def step(
                state: StepState,
            ) -> StepStream:
                yield state.updating(context=context)

        return cls(step)

    @classmethod
    def mutating_context(
        cls,
        mutation: StepContextMutating,
    ) -> Self:
        """Create a step applying an async context mutation.

        Parameters
        ----------
        mutation : StepContextMutating
            Async callable transforming the current ``ModelContext``.

        Returns
        -------
        Self
            A step that replaces context with the mutation result.

        Notes
        -----
        Rationale: centralizes context rewrite logic in explicit, testable
        callables.
        """

        async def step(
            state: StepState,
        ) -> StepStream:
            yield state.updating(
                context=await mutation(state.context),
            )

        return cls(step)

    @classmethod
    def restoring_state(
        cls,
        restoring: StepStateRestoring | StepState,
    ) -> Self:
        """Create a step replacing the current state with a stored ``StepState``.

        Parameters
        ----------
        restoring : StepStateRestoring | StepState
            Stored state snapshot or async callable returning one.

        Returns
        -------
        Self
            A step emitting the restored state as its result.

        Notes
        -----
        When ``restoring`` is callable, it is evaluated inside the
        ``"step.restore"`` context scope before the restored state is emitted.
        """

        if isinstance(restoring, StepState):

            async def step(
                state: StepState,
            ) -> StepStream:
                yield restoring

        else:

            async def step(
                state: StepState,
            ) -> StepStream:
                async with ctx.scope("step.restore"):
                    yield await restoring()

        return cls(step)

    @classmethod
    def preserving_state(
        cls,
        preserving: StepStatePreserving,
    ) -> Self:
        """Create a step storing the current whole ``StepState`` in storage.

        Parameters
        ----------
        preserving : StepStatePreserving
            Storage sink used to persist current step state.

        Returns
        -------
        Self
            A step storing the current state without modifying it.
        """

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.preserve"):
                await preserving(state)
                return  # do not emit anything

            yield  # converts function to AsyncGenerator

        return cls(step)

    @classmethod
    def appending_input(
        cls,
        input: Callable[[], Coroutine[None, None, Template | Multimodal]] | Template | Multimodal,  # noqa: A002
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Append ``ModelInput`` built from static or deferred multimodal input.

        Parameters
        ----------
        input : Callable[[], Coroutine[None, None, Template | Multimodal]]
        | Template | Multimodal
            Either immediate input payload or async provider returning it.
            ``Template`` values are resolved through ``TemplatesRepository``.
        meta : Meta | MetaValues | None
            Optional metadata attached to appended model input.

        Returns
        -------
        Self
            A step that waits for input and appends it to context.

        Notes
        -----
        Rationale: normalizes interactive/user-provided input handling in step
        pipelines.
        """

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.input"):
                resolved_input: Template | Multimodal
                if isinstance(input, Template | Multimodal):
                    resolved_input = input

                else:
                    ctx.log_debug("Waiting for input...")
                    resolved_input = await input()
                    ctx.log_debug("...input provided!")

                if isinstance(resolved_input, Template):
                    ctx.record_info(
                        attributes={"input.template": resolved_input.identifier},
                    )
                    resolved_input = await TemplatesRepository.resolve(resolved_input)

                yield state.appending_context(
                    ModelInput.of(
                        MultimodalContent.of(resolved_input),
                        meta=meta,
                    ),
                )

        return cls(step)

    @classmethod
    def appending_output(
        cls,
        output: Callable[[], Coroutine[None, None, Template | Multimodal]] | Template | Multimodal,
        /,
        *,
        emitting: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Append ``ModelOutput`` built from static or deferred multimodal output.

        Parameters
        ----------
        output : Callable[[], Coroutine[None, None, Template | Multimodal]]
        | Template | Multimodal
            Either immediate output payload or async provider returning it.
            ``Template`` values are resolved through ``TemplatesRepository``.
        emitting : bool
            When ``True``, emits each output part before appending the
            aggregated ``ModelOutput`` to context.
        meta : Meta | MetaValues | None
            Optional metadata attached to appended model output.

        Returns
        -------
        Self
            A step that records externally provided output in context.

        Notes
        -----
        Rationale: supports human-in-the-loop and external system output
        injection with consistent context updates.
        """

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.output"):
                resolved_output: Template | Multimodal
                if isinstance(output, Template | Multimodal):
                    resolved_output = output

                else:
                    ctx.log_debug("Waiting for output...")
                    resolved_output = await output()
                    ctx.log_debug("...output provided!")

                if isinstance(resolved_output, Template):
                    ctx.record_info(
                        attributes={"output.template": resolved_output.identifier},
                    )
                    resolved_output = await TemplatesRepository.resolve(resolved_output)

                output_content: MultimodalContent = MultimodalContent.of(resolved_output)
                if emitting:
                    for part in output_content.parts:
                        yield part

                yield state.appending_context(
                    ModelOutput.of(
                        output_content,
                        meta=meta,
                    )
                )

        return cls(step)

    @classmethod
    def sequence(
        cls,
        *steps: Self | StepExecuting,
    ) -> Self:
        """Compose steps into a sequential pipeline.

        Parameters
        ----------
        *steps : Self | StepExecuting
            Steps or raw step callables executed in provided order.

        Returns
        -------
        Self
            A step executing each stage sequentially, or ``Step.noop`` for an
            empty input.

        Notes
        -----
        Rationale: provides explicit deterministic ordering for state
        transformations.
        """
        if not steps:
            return cls.noop

        executions: Sequence[StepExecuting] = tuple(
            step._executing if isinstance(step, Step) else step for step in steps
        )

        async def step(
            state: StepState,
        ) -> StepStream:
            for execution in executions:
                async for chunk in execution(state=state):
                    if isinstance(chunk, StepState):
                        state = chunk

                    else:
                        yield chunk

            yield state

        return cls(step)

    @classmethod
    def loop(
        cls,
        *steps: Self | StepExecuting,
        condition: StepLoopConditionVerifying,
    ) -> Self:
        """Compose steps in a loop controlled by an async loop condition.

        Parameters
        ----------
        *steps : Self | StepExecuting
            Steps or raw step callables executed in each iteration.
        condition : StepLoopConditionVerifying
            Async predicate deciding whether next iteration should run.

        Returns
        -------
        Self
            A looping step, or ``Step.noop`` when no steps are provided.

        Notes
        -----
        Rationale: enables iterative workflows while preserving structured
        context scoping per iteration.
        """
        if not steps:
            return cls.noop

        executions: Sequence[StepExecuting] = tuple(
            step._executing if isinstance(step, Step) else step for step in steps
        )

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.loop"):
                iteration: int = 0

                while await condition(
                    state=state,
                    iteration=iteration,
                ):
                    async with ctx.scope(f"step.loop.iteration_{iteration}"):
                        for execution in executions:
                            async for chunk in execution(state=state):
                                if isinstance(chunk, StepState):
                                    state = chunk

                                else:
                                    yield chunk

                        yield state
                        iteration += 1

        return cls(step)

    @classmethod
    def concurrent(
        cls,
        *steps: Self | StepExecuting,
        merge: StepMerging,
    ) -> Self:
        """Run branch steps concurrently and merge resulting states.

        Parameters
        ----------
        *steps : Self | StepExecuting
            Branch steps or raw step callables.
        merge : StepMerging
            Async callable receiving all completed branch states and returning
            the merged state.

        Returns
        -------
        Self
            A concurrent branch step, or ``Step.noop`` for empty branches.
        """
        if not steps:
            return cls.noop

        executions: Sequence[StepExecuting] = tuple(
            step._executing if isinstance(step, Step) else step for step in steps
        )

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.concurrent"):
                output_stream: AsyncStream[StepOutputChunk] = AsyncStream()

                async def branch(
                    state: StepState,
                    execution: StepExecuting,
                ) -> StepState:
                    async with ctx.scope("step.concurrent.branch"):
                        async for chunk in execution(state=state):
                            if isinstance(chunk, StepState):
                                state = chunk

                            else:
                                await output_stream.send(chunk)

                        return state

                async with ContextTaskGroup():  # local task group for more granular management
                    branches: Sequence[Task[StepState]] = [
                        ctx.spawn(branch, state, execution) for execution in executions
                    ]

                    async def merge_branches() -> StepState:
                        try:
                            await wait(
                                branches,
                                return_when=ALL_COMPLETED,
                            )
                            output_stream.finish()

                        except BaseException as exc:
                            output_stream.finish(exc)
                            raise  # reraise original

                        return await merge(branches=(branch.result() for branch in branches))

                    merged: Task[StepState] = ctx.spawn(merge_branches)
                    async for chunk in output_stream:
                        yield chunk

                    yield await merged

        return cls(step)

    @classmethod
    def generating_completion(  # noqa: C901
        cls,
        /,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | ModelTools | Iterable[ModelToolSpecification] = ModelTools.none,
        input: Template | Multimodal | None = None,  # noqa: A002
        output: ModelOutputSelection = "auto",
        **extra: Any,
    ) -> Self:
        """Create a step that runs model completion and appends output context.

        Parameters
        ----------
        instructions : Template | ModelInstructions
            Static instructions string or template resolved at runtime.
        tools : Toolbox | ModelTools | Iterable[ModelToolSpecification]
            Toolbox, explicit ``ModelTools``, or tool specifications exposed to
            the completion call.
        input : Template | Multimodal | None = None
            Optional input content appended to context before generating completion.
        output : ModelOutputSelection
            Output selection mode passed to the model backend.
        **extra : Any
            Additional model-provider options forwarded to completion.

        Returns
        -------
        Self
            A step that appends a ``ModelOutput`` built from streamed parts.

        Raises
        ------
        Exception
            Propagates provider/model failures raised by completion execution.

        Notes
        -----
        Rationale: encapsulates output assembly (content, reasoning, tool
        requests) in one reusable, strongly typed step.
        """

        async def step(  # noqa: C901, PLR0912
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.completion"):
                if isinstance(instructions, Template):
                    ctx.record_info(attributes={"instructions.template": instructions.identifier})
                    resolved_instructions: str = await TemplatesRepository.resolve_str(instructions)

                else:
                    resolved_instructions = instructions

                if isinstance(input, Template):
                    ctx.record_info(attributes={"input.template": input.identifier})
                    state = state.appending_context(
                        ModelInput.of(await TemplatesRepository.resolve(input))
                    )

                elif input is not None:
                    state = state.appending_context(ModelInput.of(MultimodalContent.of(input)))

                model_tools: ModelTools
                if isinstance(tools, Toolbox):
                    model_tools = tools.model_tools(iteration=0)

                elif isinstance(tools, ModelTools):
                    model_tools = tools

                else:
                    model_tools = ModelTools.of(*tools)

                content_accumulator: MutableSequence[MultimodalContentPart] = []
                reasoning_accumulator: MutableSequence[ModelReasoningChunk] = []
                output_accumulator: MutableSequence[ModelOutputBlock] = []

                async for chunk in GenerativeModel.completion(
                    instructions=resolved_instructions,
                    tools=model_tools,
                    context=state.context,
                    output=output,
                    **extra,
                ):
                    yield chunk

                    if isinstance(chunk, ModelReasoningChunk):
                        if content_accumulator:
                            output_accumulator.append(MultimodalContent.of(*content_accumulator))
                            content_accumulator.clear()

                        reasoning_accumulator.append(chunk)

                    elif isinstance(chunk, ModelToolRequest):
                        if content_accumulator:
                            output_accumulator.append(MultimodalContent.of(*content_accumulator))
                            content_accumulator.clear()

                        if reasoning_accumulator:
                            output_accumulator.append(ModelReasoning.of(reasoning_accumulator))
                            reasoning_accumulator.clear()

                        output_accumulator.append(chunk)

                    else:
                        if reasoning_accumulator:
                            output_accumulator.append(ModelReasoning.of(reasoning_accumulator))
                            reasoning_accumulator.clear()

                        content_accumulator.append(chunk)

                if content_accumulator:
                    output_accumulator.append(MultimodalContent.of(*content_accumulator))

                if reasoning_accumulator:
                    output_accumulator.append(ModelReasoning.of(reasoning_accumulator))

                yield state.appending_context(ModelOutput.of(*output_accumulator))

        return cls(step)

    @classmethod
    def handling_tools(
        cls,
        tools: Toolbox | Sequence[Tool],
        /,
    ) -> Self:
        """Create a step handling tool requests from the latest model output.

        Parameters
        ----------
        tools : Toolbox | Sequence[Tool]
            Toolbox instance or tool sequence used to resolve requests.

        Returns
        -------
        Self
            A step executing pending tool requests and appending responses.

        Raises
        ------
        Exception
            Propagates tool-handler failures.

        Notes
        -----
        Rationale: keeps tool-dispatch semantics centralized and consistent
        across pipelines.
        """
        toolbox: Toolbox = Toolbox.of(tools)

        async def step(
            state: StepState,
        ) -> StepStream:
            if not state.context:
                yield state
                return  # empty context

            last_output: ModelInput | ModelOutput = state.context[-1]
            if not isinstance(last_output, ModelOutput):
                yield state
                return  # ending with input

            tool_requests: Sequence[ModelToolRequest] = last_output.tool_requests
            if not tool_requests:
                yield state
                return  # no tools requested

            async with ctx.scope("step.tools.handling"):
                responses: MutableSequence[ModelToolResponse] = []
                tools_output_accumulator: MutableSequence[MultimodalContentPart] = []
                async for chunk in toolbox.handle(*tool_requests):
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
                    yield state.appending_context(
                        ModelInput.of(*responses),
                        ModelOutput.of(MultimodalContent.of(*tools_output_accumulator)),
                    )

                else:  # regular tools result
                    yield state.appending_context(ModelInput.of(*responses))

        return cls(step)

    @classmethod
    def looping_completion(  # noqa: C901, PLR0915
        cls,
        /,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        input: Template | Multimodal | None = None,  # noqa: A002
        output: ModelOutputSelection = "auto",
        **extra: Any,
    ) -> Self:
        """Create a step looping between completion and tool execution.

        The step repeatedly runs completion, executes requested tools, and
        appends responses. The loop ends when completion returns no tool
        requests, or earlier when any tool response uses ``handling="output"``
        and output content is appended directly.

        Parameters
        ----------
        instructions : Template | ModelInstructions
            Static instructions string or template resolved at runtime.
        tools : Toolbox | Iterable[Tool]
            Toolbox or tool sequence available during loop iterations.
        input : Template | Multimodal | None = None
            Optional input content appended to context before generating completion.
        output : ModelOutputSelection
            Output selection mode passed to model completion.
        **extra : Any
            Additional provider options forwarded to completion.

        Returns
        -------
        Self
            A step implementing iterative completion-with-tools behavior.

        Raises
        ------
        Exception
            Propagates model and tool execution failures.

        Notes
        -----
        Rationale: packages completion-and-tool loop logic into a reusable
        typed primitive.
        """
        toolbox: Toolbox = Toolbox.of(tools)

        async def step(  # noqa: C901, PLR0912, PLR0915
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.completion.loop"):
                if isinstance(instructions, Template):
                    ctx.record_info(attributes={"instructions.template": instructions.identifier})
                    resolved_instructions: str = await TemplatesRepository.resolve_str(instructions)

                else:
                    resolved_instructions = instructions

                if isinstance(input, Template):
                    ctx.record_info(attributes={"input.template": input.identifier})
                    state = state.appending_context(
                        ModelInput.of(await TemplatesRepository.resolve(input))
                    )

                elif input is not None:
                    state = state.appending_context(ModelInput.of(MultimodalContent.of(input)))

                iteration: int = 0
                while True:  # loop until we get ModelOutput without tools
                    async with ctx.scope(f"step.completion.loop.iteration_{iteration}"):
                        model_output: ModelOutput
                        ctx.log_debug("Generating completion...")
                        content_accumulator: MutableSequence[MultimodalContentPart] = []
                        reasoning_accumulator: MutableSequence[ModelReasoningChunk] = []
                        output_accumulator: MutableSequence[ModelOutputBlock] = []

                        async for chunk in GenerativeModel.completion(
                            instructions=resolved_instructions,
                            tools=toolbox.model_tools(iteration=iteration),
                            context=state.context,
                            output=output,
                            **extra,
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
                        async for chunk in toolbox.handle(*tool_requests):
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
                            yield state.appending_context(
                                ModelInput.of(*responses),
                                ModelOutput.of(MultimodalContent.of(*tools_output_accumulator)),
                            )
                            break  # end of loop

                        else:  # regular tools result
                            state = state.appending_context(
                                ModelInput.of(*responses),
                            )
                            yield state
                            iteration += 1  # continue next iteration

        return cls(step)

    @classmethod
    def selection(
        cls,
        selecting: StepSelecting,
    ) -> Self:
        """Create a step that selects and executes another step at runtime.

        Parameters
        ----------
        selecting : StepSelecting
            Async callable receiving the current ``StepState`` and returning
            the ``Step`` to execute for that state.

        Returns
        -------
        Self
            A step delegating execution to the step returned by ``selecting``.

        Notes
        -----
        Rationale: supports state-driven branching when the chosen behavior
        must remain composable as a regular ``Step``.
        """

        async def step(
            state: StepState,
        ) -> StepStream:
            selection: Step = await selecting(state=state)

            async for chunk in selection._executing(state):
                yield chunk

        return cls(step)

    __slots__ = ("_executing",)

    def __init__(
        self,
        executing: StepExecuting,
        /,
    ) -> None:
        """Initialize ``Step`` with a low-level execution coroutine.

        Parameters
        ----------
        executing : StepExecuting
            Async callable receiving ``StepState`` and yielding state updates
            and/or output chunks.
        """
        self._executing: StepExecuting
        object.__setattr__(
            self,
            "_executing",
            executing,
        )

    def with_ctx(
        self,
        *ctx_state: State,
        disposables: Collection[Disposable] = (),
    ) -> Self:
        """Bind additional scoped context state and disposables to execution.

        Parameters
        ----------
        *ctx_state : State
            State instances injected via ``ctx.updating`` for this step.
        disposables : Collection[Disposable]
            Disposable resources entered for the step lifetime.

        Returns
        -------
        Self
            A wrapped step with additional scoped runtime context, or ``self``
            when no context state and disposables are provided.

        Notes
        -----
        Rationale: allows local dependency injection without introducing global
        mutable state.
        """
        executing: StepExecuting = self._executing

        if ctx_state:
            if disposables:

                async def step(
                    state: StepState,
                ) -> StepStream:
                    async with Disposables(disposables) as disposable_state:
                        with ctx.updating(*disposable_state, *ctx_state):
                            async for chunk in executing(state=state):
                                yield chunk
            else:

                async def step(
                    state: StepState,
                ) -> StepStream:
                    with ctx.updating(*ctx_state):
                        async for chunk in executing(state=state):
                            yield chunk

        elif disposables:

            async def step(
                state: StepState,
            ) -> StepStream:
                async with ctx.disposables(*disposables):
                    async for chunk in executing(state=state):
                        yield chunk

        else:
            return self  # nothing to change...

        return self.__class__(step)

    def with_retry(  # noqa: C901
        self,
        *,
        limit: int = 1,
        delay: Callable[[int, Exception], float] | float | None = None,
        catching: Callable[[Exception], bool] | type[Exception] = Exception,
    ) -> Self:
        """Wrap step execution with retry policy.

        Parameters
        ----------
        limit : int
            Maximum number of retries after the initial failed attempt.
        delay : Callable[[int, Exception], float] | float | None
            Static delay or callback computing delay per failed attempt.
        catching : Callable[[Exception], bool] | type[Exception]
            Exception type or predicate deciding whether failure is retryable.

        Returns
        -------
        Self
            A step retried according to provided policy.

        Notes
        -----
        Rationale: adds resilience declaratively while preserving step
        composability.
        """
        executing: StepExecuting = self._executing
        catch_check: Callable[[Exception], bool]
        if isinstance(catching, type):

            def check(exc: Exception) -> bool:
                return isinstance(exc, catching)

            catch_check = check

        else:
            catch_check = catching

        async def step(
            state: StepState,
        ) -> StepStream:
            attempt: int = 0
            while True:
                try:
                    async for chunk in executing(state=state):
                        yield chunk

                        if isinstance(chunk, StepState):
                            state = chunk  # update local state in case of retry

                except Exception as exc:
                    if attempt < limit and catch_check(exc):
                        attempt += 1
                        ctx.log_error(
                            "Attempting to retry step which failed due to an error",
                            exception=exc,
                        )

                        match delay:
                            case None:
                                continue

                            case float(strict) | int(strict):
                                await sleep(float(strict))

                            case make_delay:
                                await sleep(make_delay(attempt, exc))  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]

                    else:
                        raise  # uncatched exception

                else:
                    return  # end the loop

        return self.__class__(step)

    def with_fallback(
        self,
        fallback: Self,
        *,
        catching: set[type[Exception]] | tuple[type[Exception], ...] | type[Exception] = Exception,
    ) -> Self:
        """Use a fallback step when selected exceptions are raised.

        Parameters
        ----------
        fallback : Self
            Step executed when primary execution fails with a caught exception.
        catching : set[type[Exception]] | tuple[type[Exception], ...] | type[Exception]
            Exception type(s) that trigger fallback execution.

        Returns
        -------
        Self
            A step with fallback behavior.

        Raises
        ------
        Exception
            Re-raises exceptions not matched by ``catching``.

        Notes
        -----
        Rationale: enables controlled degradation while preserving explicit
        error boundaries.
        """
        executing: StepExecuting = self._executing
        fallback_executing: StepExecuting = fallback._executing
        exceptions: Collection[type[Exception]] = (
            catching if isinstance(catching, set | tuple) else {catching}
        )

        async def step(
            state: StepState,
        ) -> StepStream:
            try:
                async for chunk in executing(state=state):
                    yield chunk

                    if isinstance(chunk, StepState):
                        state = chunk  # update local state in case of fallback

            except Exception as exc:
                if any(isinstance(exc, exception) for exception in exceptions):
                    ctx.log_info(f"Using fallback Step for {type(exc)}")
                    async for chunk in fallback_executing(state=state):
                        yield chunk

                else:
                    raise  # reraise original

        return self.__class__(step)

    def with_isolated_context(
        self,
        context: Callable[[], Coroutine[None, None, ModelContext]] | ModelContext = (),
    ) -> Self:
        """Execute this step against an isolated context snapshot.

        Parameters
        ----------
        context : Callable[[], Coroutine[None, None, ModelContext]] | ModelContext
            Replacement context, or async provider returning one, used only for
            the wrapped execution.

        Returns
        -------
        Self
            A wrapped step that restores the original context on emitted state
            updates.

        Notes
        -----
        Rationale: allows temporary context substitution without letting
        wrapped execution mutate the caller-visible context.
        """
        executing: StepExecuting = self._executing

        if iscoroutinefunction(context):

            async def step(
                state: StepState,
            ) -> StepStream:
                async for chunk in executing(state=state.updating(context=await context())):
                    if isinstance(chunk, StepState):
                        yield chunk.updating(context=state.context)

                    else:
                        yield chunk

        else:

            async def step(
                state: StepState,
            ) -> StepStream:
                async for chunk in executing(state=state.updating(context=context)):
                    if isinstance(chunk, StepState):
                        yield chunk.updating(context=state.context)

                    else:
                        yield chunk

        return self.__class__(step)

    def with_volatile_context(self) -> Self:
        """Discard context changes produced by this step.

        Returns
        -------
        Self
            A wrapped step restoring original context after execution.

        Notes
        -----
        Rationale: isolates context mutations when only side effects or emitted
        output should survive.
        """
        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            async for chunk in executing(state=state):
                if isinstance(chunk, StepState):
                    yield chunk.updating(context=state.context)

                else:
                    yield chunk

        return self.__class__(step)

    def with_volatile_tools(self) -> Self:
        """Discard tool-bearing context elements produced by this step.

        Returns
        -------
        Self
            A wrapped step removing context elements whose
            ``contains_tools`` flag is set after execution.

        Notes
        -----
        Rationale: keeps transient tool definitions or requests from leaking
        into subsequent step context while preserving other state updates.
        """
        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            async for chunk in executing(state=state):
                if isinstance(chunk, StepState):
                    updated: ModelContext = tuple(  # remove context elements containing tools
                        element for element in chunk.context if not element.contains_tools
                    )
                    if chunk.context != updated:
                        yield chunk.updating(context=updated)

                else:
                    yield chunk

        return self.__class__(step)

    def with_condition(  # noqa: C901
        self,
        condition: StepConditionVerifying | bool,
        /,
        *,
        alternative: Self | None = None,
    ) -> Self:
        """Execute this step conditionally, optionally with an alternative.

        Parameters
        ----------
        condition : StepConditionVerifying | bool
            Static boolean or async predicate evaluated against current state.
        alternative : Self | None
            Optional step executed when condition evaluates to ``False``.

        Returns
        -------
        Self
            A conditionally executed step wrapper.

        Notes
        -----
        Rationale: enables branch logic without breaking fluent composition.
        """
        executing: StepExecuting = self._executing
        alternative_executing: StepExecuting | None = (
            alternative._executing if alternative else None
        )

        if isinstance(condition, bool):
            if alternative_executing is None:

                def step(
                    state: StepState,
                ) -> StepStream:
                    if condition:
                        return executing(state=state)

                    else:
                        return _noop_stream()

            else:

                def step(
                    state: StepState,
                ) -> StepStream:
                    if condition:
                        return executing(state=state)

                    else:
                        return alternative_executing(state=state)

        elif alternative_executing is None:

            async def step(
                state: StepState,
            ) -> StepStream:
                if await condition(state=state):
                    async for chunk in executing(state=state):
                        yield chunk

        else:

            async def step(
                state: StepState,
            ) -> StepStream:
                if await condition(state=state):
                    async for chunk in executing(state=state):
                        yield chunk

                else:
                    async for chunk in alternative_executing(state=state):
                        yield chunk

        return self.__class__(step)

    def with_suppressed_output(self) -> Self:
        """Suppress all emitted output from this step.

        Returns
        -------
        Self
            A wrapped step that keeps state changes but discards emissions.

        Notes
        -----
        Rationale: useful when only state mutation is needed from a noisy step.
        """
        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            async for chunk in executing(state=state):
                if not isinstance(chunk, StepState):
                    continue  # skip output chunks

                yield chunk  # pass only state updates

        return self.__class__(step)

    def with_context_evaluation(
        self,
        evaluator: PreparedEvaluatorScenario[ModelContext] | PreparedEvaluator[ModelContext],
        /,
        *,
        raise_on_failure: bool = False,
    ) -> Self:
        """Evaluate current context before executing this step.

        Parameters
        ----------
        evaluator : PreparedEvaluatorScenario[ModelContext] | PreparedEvaluator[ModelContext]
            Prepared evaluator checking the current model context.
        raise_on_failure : bool
            Whether a failed evaluation should raise ``StepException``.
            When ``False``, evaluation results are reported by evaluator side
            effects and step execution continues.

        Returns
        -------
        Self
            A wrapped step that always evaluates context before execution.

        Raises
        ------
        StepException
            Raised when evaluation fails and fail-fast mode is enabled.

        Notes
        -----
        Rationale: integrates quality/safety checks directly into step
        execution flow.
        """
        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.evaluation.context"):
                result: EvaluatorScenarioResult | EvaluatorResult = await evaluator(state.context)

                if raise_on_failure and not result.passed:
                    raise StepException(
                        "Context evaluation failed",
                        state=state,
                        meta={
                            "performance": result.performance,
                            "report": result.report(detailed=__debug__),
                        },
                    )

            async for chunk in executing(state=state):
                yield chunk

        return self.__class__(step)

    def with_output_evaluation(
        self,
        evaluator: PreparedEvaluatorScenario[Sequence[StepOutputChunk]]
        | PreparedEvaluator[Sequence[StepOutputChunk]],
        /,
        *,
        raise_on_failure: bool = False,
    ) -> Self:
        """Evaluate emitted output chunks incrementally during execution.

        Parameters
        ----------
        evaluator : PreparedEvaluatorScenario[Sequence[StepOutputChunk]]
        | PreparedEvaluator[Sequence[StepOutputChunk]]
            Prepared evaluator checking the accumulated output.
        raise_on_failure : bool
            Whether a failed evaluation should raise ``StepException``.
            When ``False``, evaluation results are reported by evaluator side
            effects and step execution continues.

        Returns
        -------
        Self
            A wrapped step that evaluates currently accumulated output chunks.

        Raises
        ------
        StepException
            Raised when evaluation fails and fail-fast mode is enabled.

        Notes
        -----
        Rationale: integrates quality/safety checks directly into step
        execution flow.
        """
        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            accumulator: MutableSequence[StepOutputChunk] = []
            async for chunk in executing(state=state):
                yield chunk

                if isinstance(chunk, StepState):
                    state = chunk  # update local state
                    continue  # evaluate only output

                accumulator.append(chunk)

                async with ctx.scope("step.evaluation.output"):
                    result: EvaluatorScenarioResult | EvaluatorResult = await evaluator(accumulator)

                    if raise_on_failure and not result.passed:
                        raise StepException(
                            "Output evaluation failed",
                            state=state,
                            meta={
                                "performance": result.performance,
                                "report": result.report(detailed=__debug__),
                            },
                        )

        return self.__class__(step)

    @overload
    async def run(
        self,
        state: StepState,
        /,
    ) -> MultimodalContent: ...

    @overload
    async def run(
        self,
        context: ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> MultimodalContent: ...

    async def run(
        self,
        context: StepState | ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> MultimodalContent:
        """Execute step and collect emitted multimodal content parts.

        Parameters
        ----------
        context : StepState | ModelContext
            Initial model context used to construct ``StepState`` or state directly.
        *artifacts : State
            Artifact states stored under their class names.
        **keyed_artifacts : State
            Artifact states stored under explicit keys.

        Returns
        -------
        MultimodalContent
            Collected emitted multimodal parts in emission order.
            Non-multimodal chunks are ignored.

        Notes
        -----
        Rationale: provides a convenience API for consumers interested only in
        user-visible emitted content.
        """
        accumulator: MutableSequence[MultimodalContentPart] = []

        stream: StepStream
        if isinstance(context, StepState):
            assert not artifacts  # nosec: B101
            assert not keyed_artifacts  # nosec: B101
            stream = self._executing(
                state=context,
            )

        else:
            stream = self._executing(
                state=StepState.of(
                    context,
                    *artifacts,
                    **keyed_artifacts,
                ),
            )

        async for chunk in stream:
            if isinstance(chunk, MultimodalContentPart):
                accumulator.append(chunk)

        return MultimodalContent.of(*accumulator)

    @overload
    async def process(
        self,
        state: StepState,
        /,
    ) -> StepState: ...

    @overload
    async def process(
        self,
        context: ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> StepState: ...

    async def process(
        self,
        context: StepState | ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> StepState:
        """Execute step and return the final ``StepState``.

        Parameters
        ----------
        context : StepState | ModelContext
            Initial model context used to construct ``StepState`` or state directly.
        *artifacts : State
            Artifact states stored under their class names.
        **keyed_artifacts : State
            Artifact states stored under explicit keys.

        Returns
        -------
        StepState
            Final state after executing the step.

        Notes
        -----
        Rationale: provides a convenience API for consumers interested only in
        final state transformation.
        """
        state: StepState

        if isinstance(context, StepState):
            assert not artifacts  # nosec: B101
            assert not keyed_artifacts  # nosec: B101
            state = context

        else:
            state = StepState.of(
                context,
                *artifacts,
                **keyed_artifacts,
            )

        async for chunk in self._executing(state=state):
            if isinstance(chunk, StepState):
                state = chunk

        return state

    @overload
    def stream(
        self,
        state: StepState,
        /,
    ) -> AsyncIterable[StepOutputChunk]: ...

    @overload
    def stream(
        self,
        context: ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> AsyncIterable[StepOutputChunk]: ...

    async def stream(
        self,
        context: StepState | ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> AsyncIterable[StepOutputChunk]:
        """Execute step and stream emitted non-state output chunks.

        Parameters
        ----------
        context : StepState | ModelContext
            Initial model context used to construct ``StepState`` or state directly.
        *artifacts : State
            Artifact states stored under their class names.
        **keyed_artifacts : State
            Artifact states stored under explicit keys.

        Returns
        -------
        AsyncIterable[StepOutputChunk]
            Async stream yielding emitted output chunks until completion.

        Raises
        ------
        BaseException
            Propagates exceptions raised by the wrapped step.

        Notes
        -----
        Rationale: exposes streaming execution for incremental consumers such as
        UIs and realtime pipelines.
        """

        stream: StepStream
        if isinstance(context, StepState):
            assert not artifacts  # nosec: B101
            assert not keyed_artifacts  # nosec: B101
            stream = self._executing(state=context)

        else:
            stream = self._executing(
                state=StepState.of(
                    context,
                    *artifacts,
                    **keyed_artifacts,
                ),
            )

        async for chunk in stream:
            if isinstance(chunk, StepState):
                continue  # skip state updates

            yield chunk  # provide the rest

    def __aiter__(self) -> AsyncIterator[StepOutputChunk]:
        return aiter(self.stream())

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


def step(
    processing: StepProcessing,
    /,
) -> Step:
    """Adapt a state-only processor into a full ``Step``.

    Parameters
    ----------
    processing : StepProcessing
        Async callable transforming ``StepState`` without emitting output.

    Returns
    -------
    Step
        Step wrapper invoking ``processing`` and forwarding state updates.

    Notes
    -----
    Rationale: offers a minimal bridge from simple state transformations to the
    richer step execution protocol.
    """

    async def executing(
        state: StepState,
    ) -> StepStream:
        yield await processing(state)

    return Step(executing)


async def _noop_stream() -> StepStream:
    return  # do not emit anything
    yield  # converts to AsyncGenerator


@Step
async def _noop(
    state: StepState,
) -> StepStream:
    return  # do not emit anything
    yield  # converts to AsyncGenerator


Step.noop = _noop


@runtime_checkable
class StepSelecting(Protocol):
    """Protocol for runtime step selection based on the current state."""

    async def __call__(
        self,
        state: StepState,
    ) -> Step: ...
