from asyncio import sleep, timeout
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from time import monotonic
from typing import Any, ClassVar, NoReturn, Protocol, Self, cast, final, overload, runtime_checkable

from haiway import (
    Disposable,
    Disposables,
    Meta,
    MetaValues,
    State,
    ctx,
    stream_concurrently,
)

from draive._constant import class_constant
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
    ModelOutputChunk,
    ModelOutputSelection,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    ModelToolSpecification,
    model_output_blocks,
)
from draive.models.types import ModelOutputStream
from draive.multimodal import (
    ArtifactContent,
    Multimodal,
    MultimodalContent,
    MultimodalContentPart,
    Template,
    TemplatesRepository,
)
from draive.steps.decoding import decode_content
from draive.steps.state import StepState, artifacts_mapping
from draive.steps.types import (
    StepConditionVerifying,
    StepContextMutating,
    StepException,
    StepExecuting,
    StepLoopConditionVerifying,
    StepMerging,
    StepOutputChunk,
    StepProcessing,
    StepRetriesExhausted,
    StepStatePreserving,
    StepStateRestoring,
    StepStream,
    StepTimeout,
)
from draive.tools import Tool, Toolbox
from draive.utils import ProcessingEvent

__all__ = (
    "Step",
    "StepSelecting",
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

    noop: ClassVar[Step] = class_constant(lambda: _noop)

    @classmethod
    def emitting(
        cls,
        *parts: StepOutputChunk | Multimodal,
    ) -> Self:
        """Create a step that emits fixed output parts and leaves state unchanged.

        Parameters
        ----------
        *parts : StepOutputChunk | Multimodal
            Output parts to emit in order. Multimodal values - including plain
            ``str`` and ``MultimodalContent`` containers - are normalized into
            individual content parts, since only those are recognized as
            emitted output by stream consumers.

        Returns
        -------
        Self
            A step emitting the provided parts, or ``Step.noop`` when nothing
            would be emitted.

        Notes
        -----
        Rationale: useful for injecting deterministic stream output into a
        processing pipeline.
        """
        if not parts:
            return cast(Self, cls.noop)

        emitted: MutableSequence[StepOutputChunk] = []
        for part in parts:
            if isinstance(
                part,
                ModelReasoningChunk | ModelToolRequest | ModelToolResponse | ProcessingEvent,
            ):
                emitted.append(part)

            else:  # normalize multimodal forms into individual content parts
                emitted.extend(MultimodalContent.of(part).parts)

        if not emitted:
            return cast(Self, cls.noop)

        async def step(
            state: StepState,
        ) -> StepStream:
            for chunk in emitted:
                yield chunk

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

        Raises
        ------
        ValueError
            If provided artifacts resolve to the same key.

        Notes
        -----
        Rationale: enables declarative artifact injection without custom step
        implementations.
        """
        if not artifacts and not keyed_artifacts:
            return cast(Self, cls.noop)

        merged_artifacts: Mapping[str, ArtifactContent] = artifacts_mapping(
            artifacts,
            keyed_artifacts,
        )

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
    def removing_artifacts(
        cls,
        *keys: str | type[State],
    ) -> Self:
        """Create a step that removes artifacts from ``StepState``.

        Parameters
        ----------
        *keys : str | type[State]
            Artifact keys to remove. A state type selects the artifact stored
            under its class name. Missing keys are ignored.

        Returns
        -------
        Self
            A step removing selected artifacts, or ``Step.noop`` when no keys
            are provided.
        """
        if not keys:
            return cast(Self, cls.noop)

        async def step(
            state: StepState,
        ) -> StepStream:
            yield state.removing_artifacts(*keys)

        return cls(step)

    @classmethod
    def decoding_artifact[Artifact: State](
        cls,
        artifact: type[Artifact],
        /,
        *,
        key: str | None = None,
    ) -> Self:
        """Create a step decoding the latest model output into a typed artifact.

        Intended to follow a completion step requesting structured output, i.e.
        ``Step.looping_completion(output=Report)``. A direct JSON artifact of
        the output is preferred, otherwise the JSON text of the output is decoded.

        Parameters
        ----------
        artifact : type[Artifact]
            State type to decode.
        key : str | None
            Artifact key. Defaults to the class name of ``artifact``.

        Returns
        -------
        Self
            A step storing the decoded artifact in ``StepState``.

        Raises
        ------
        StepException
            If context does not end with a model output or decoding fails.
        """

        async def step(
            state: StepState,
        ) -> StepStream:
            if not state.context or not isinstance(state.context[-1], ModelOutput):
                raise StepException(
                    f"Decoding {artifact.__name__} requires context ending with model output",
                    state=state,
                )

            try:
                decoded: Artifact = decode_content(state.context[-1].content, artifact)

            except Exception as exc:
                raise StepException(
                    f"Failed to decode {artifact.__name__} from model output",
                    state=state,
                ) from exc

            yield state.updating_artifacts(**{key or artifact.__name__: decoded})

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
            return cast(Self, cls.noop)

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

        if callable(context):

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
            return cast(Self, cls.noop)

        executions: Sequence[StepExecuting] = tuple(
            step._executing if isinstance(step, Step) else step for step in steps
        )

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.sequence"):
                for execution in executions:
                    execution_stream: StepStream = execution(state=state)
                    try:
                        async for chunk in execution_stream:
                            if isinstance(chunk, StepState):
                                state = chunk

                            else:
                                yield chunk

                    finally:
                        await execution_stream.aclose()

                yield state

        return cls(step)

    @classmethod
    def loop(
        cls,
        *steps: Self | StepExecuting,
        condition: StepLoopConditionVerifying,
        max_iterations: int | None = 32,
    ) -> Self:
        """Compose steps in a loop controlled by an async loop condition.

        Parameters
        ----------
        *steps : Self | StepExecuting
            Steps or raw step callables executed in each iteration.
        condition : StepLoopConditionVerifying
            Async predicate deciding whether next iteration should run.
        max_iterations : int | None
            Safety limit of iterations, ``None`` disables it. Defaults to 32.

        Returns
        -------
        Self
            A looping step, or ``Step.noop`` when no steps are provided.

        Raises
        ------
        StepException
            When the condition still requests another iteration after
            ``max_iterations`` were executed.
        ValueError
            When ``max_iterations`` is lower than 1.

        Notes
        -----
        Rationale: enables iterative workflows while preserving structured
        context scoping per iteration.
        """
        _validate_max_iterations(max_iterations)
        if not steps:
            return cast(Self, cls.noop)

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
                    if max_iterations is not None and iteration >= max_iterations:
                        raise StepException(
                            f"Step loop exceeded the limit of {max_iterations} iterations",
                            state=state,
                            meta={"max_iterations": max_iterations},
                        )

                    async with ctx.scope(f"step.loop.iteration_{iteration}"):
                        for execution in executions:
                            execution_stream: StepStream = execution(state=state)
                            try:
                                async for chunk in execution_stream:
                                    if isinstance(chunk, StepState):
                                        state = chunk

                                    else:
                                        yield chunk

                            finally:
                                await execution_stream.aclose()

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
            return cast(Self, cls.noop)

        executions: Sequence[StepExecuting] = tuple(
            step._executing if isinstance(step, Step) else step for step in steps
        )

        async def step(
            state: StepState,
        ) -> StepStream:
            async with ctx.scope("step.concurrent"):
                # final states of the branches, keyed by their position
                results: MutableMapping[int, StepState] = {}

                async def branch(
                    index: int,
                    execution: StepExecuting,
                ) -> StepStream:
                    async with ctx.scope("step.concurrent.branch"):
                        branch_state: StepState = state
                        execution_stream: StepStream = execution(state=state)
                        try:
                            async for chunk in execution_stream:
                                if isinstance(chunk, StepState):
                                    branch_state = chunk

                                else:
                                    yield chunk

                        finally:
                            await execution_stream.aclose()

                        results[index] = branch_state

                # branches run concurrently, their output is delivered as it comes and
                # the first branch failure ends the rest
                branches_stream: StepStream = stream_concurrently(
                    *(branch(index, execution) for index, execution in enumerate(executions)),
                    exhaustive=True,
                )
                try:
                    async for chunk in branches_stream:
                        yield chunk

                finally:
                    await branches_stream.aclose()

                yield await merge(
                    branches=tuple(results[index] for index in range(len(executions))),
                )

        return cls(step)

    @classmethod
    def generating_completion(
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

        async def step(
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

                chunks_accumulator: MutableSequence[ModelOutputChunk] = []

                completion_stream: ModelOutputStream = GenerativeModel.completion(
                    instructions=resolved_instructions,
                    tools=model_tools,
                    context=state.context,
                    output=output,
                    **extra,
                )
                try:
                    async for chunk in completion_stream:
                        yield chunk
                        chunks_accumulator.append(chunk)

                finally:
                    await completion_stream.aclose()

                yield state.appending_context(
                    ModelOutput.of(*model_output_blocks(chunks_accumulator))
                )

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
                handling_stream: StepStream = _handling_tool_requests(
                    toolbox,
                    tool_requests,
                    state=state,
                )
                try:
                    async for chunk in handling_stream:
                        yield chunk

                finally:
                    await handling_stream.aclose()

        return cls(step)

    @classmethod
    def looping_completion(  # noqa: C901
        cls,
        /,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        input: Template | Multimodal | None = None,  # noqa: A002
        output: ModelOutputSelection = "auto",
        max_iterations: int | None = 32,
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
        max_iterations : int | None
            Safety limit of completion rounds, ``None`` disables it. Defaults to 32.
        **extra : Any
            Additional provider options forwarded to completion.

        Returns
        -------
        Self
            A step implementing iterative completion-with-tools behavior.

        Raises
        ------
        StepException
            When the model keeps requesting tools after ``max_iterations`` rounds.
        ValueError
            When ``max_iterations`` is lower than 1.
        Exception
            Propagates model and tool execution failures.

        Notes
        -----
        Rationale: packages completion-and-tool loop logic into a reusable
        typed primitive.
        """
        _validate_max_iterations(max_iterations)
        toolbox: Toolbox = Toolbox.of(tools)

        async def step(  # noqa: C901, PLR0912
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
                    if max_iterations is not None and iteration >= max_iterations:
                        raise StepException(
                            f"Completion loop exceeded the limit of {max_iterations} iterations",
                            state=state,
                            meta={"max_iterations": max_iterations},
                        )

                    async with ctx.scope(f"step.completion.loop.iteration_{iteration}"):
                        model_output: ModelOutput
                        ctx.log_debug("Generating completion...")
                        chunks_accumulator: MutableSequence[ModelOutputChunk] = []

                        completion_stream: ModelOutputStream = GenerativeModel.completion(
                            instructions=resolved_instructions,
                            tools=toolbox.model_tools(iteration=iteration),
                            context=state.context,
                            output=output,
                            **extra,
                        )
                        try:
                            async for chunk in completion_stream:
                                yield chunk
                                # TODO: start handling tool requests immediately
                                chunks_accumulator.append(chunk)

                        finally:
                            await completion_stream.aclose()

                        model_output: ModelOutput = ModelOutput.of(
                            *model_output_blocks(chunks_accumulator)
                        )

                        state = state.appending_context(model_output)
                        yield state

                        tool_requests: Sequence[ModelToolRequest] = model_output.tool_requests
                        if not tool_requests:
                            break  # end of loop

                        ctx.log_debug("...handling tool requests...")
                        handling_stream: StepStream = _handling_tool_requests(
                            toolbox,
                            tool_requests,
                            state=state,
                        )
                        try:
                            async for chunk in handling_stream:
                                if isinstance(chunk, StepState):
                                    state = chunk

                                else:
                                    yield chunk

                        finally:
                            await handling_stream.aclose()

                        yield state
                        if isinstance(state.context[-1], ModelOutput):
                            break  # tools generated output directly, end of loop

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

            selection_stream = selection._executing(state)
            try:
                async for chunk in selection_stream:
                    yield chunk

            finally:
                await selection_stream.aclose()

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
                            execution_stream = executing(state=state)
                            try:
                                async for chunk in execution_stream:
                                    yield chunk

                            finally:
                                await execution_stream.aclose()

            else:

                async def step(
                    state: StepState,
                ) -> StepStream:
                    with ctx.updating(*ctx_state):
                        execution_stream = executing(state=state)
                        try:
                            async for chunk in execution_stream:
                                yield chunk

                        finally:
                            await execution_stream.aclose()

        elif disposables:

            async def step(
                state: StepState,
            ) -> StepStream:
                async with ctx.disposables(*disposables):
                    execution_stream = executing(state=state)
                    try:
                        async for chunk in execution_stream:
                            yield chunk

                    finally:
                        await execution_stream.aclose()

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
            A step retried according to provided policy. Every attempt starts from the
            original input state, partial state emitted by a failed attempt is discarded.
            Output chunks already emitted by a failed attempt cannot be retracted.
            Failing after at least one retry raises ``StepRetriesExhausted`` chained
            to the last error, uncaught errors and failures without any retry
            are raised as they are.

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
                state_emitted: bool = False
                try:
                    execution_stream = executing(state=state)
                    try:
                        async for chunk in execution_stream:
                            yield chunk

                            if isinstance(chunk, StepState):
                                state_emitted = True

                    finally:
                        await execution_stream.aclose()

                except Exception as exc:
                    if attempt < limit and catch_check(exc):
                        attempt += 1
                        if state_emitted:
                            yield state  # discard partial state of the failed attempt

                        ctx.log_error(
                            "Attempting to retry step which failed due to an error",
                            exception=exc,
                        )

                        retry_delay: float = _retry_delay(delay, attempt, exc)
                        if retry_delay > 0:
                            await sleep(retry_delay)

                    elif attempt > 0 and catch_check(exc):
                        raise StepRetriesExhausted(
                            f"Step failed after {attempt + 1} attempts",
                            state=state,
                            attempts=attempt + 1,
                        ) from exc

                    else:
                        raise  # uncatched exception

                else:
                    return  # end the loop

        return self.__class__(step)

    def with_timeout(
        self,
        seconds: float,
        /,
    ) -> Self:
        """Limit the execution time of this step.

        Only time spent executing the step counts, time the consumer takes to
        process emitted chunks between them is excluded.

        Parameters
        ----------
        seconds : float
            Maximum execution time in seconds, must be positive.

        Returns
        -------
        Self
            A step raising ``StepTimeout`` when the limit is exceeded.

        Raises
        ------
        ValueError
            When ``seconds`` is not positive.
        """
        if seconds <= 0:
            raise ValueError("Step timeout has to be positive")

        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            remaining: float = seconds
            execution_stream: StepStream = executing(state=state)
            try:
                while True:
                    started: float = monotonic()
                    try:
                        async with timeout(max(remaining, 0)):
                            chunk: StepOutputChunk | StepState = await anext(execution_stream)

                    except StopAsyncIteration:
                        return

                    except TimeoutError as exc:
                        raise StepTimeout(
                            f"Step timed out after {seconds} seconds",
                            state=state,
                            timeout=seconds,
                        ) from exc

                    remaining -= monotonic() - started
                    if isinstance(chunk, StepState):
                        state = chunk

                    yield chunk

            finally:
                await execution_stream.aclose()

        return self.__class__(step)

    def then(
        self,
        *steps: Self | StepExecuting,
    ) -> Self:
        """Compose this step with following steps into a sequence.

        Parameters
        ----------
        *steps : Self | StepExecuting
            Steps executed after this one, in order.

        Returns
        -------
        Self
            A sequential step, equivalent to ``Step.sequence(self, *steps)``.
        """
        return self.__class__.sequence(self, *steps)

    def __rshift__(
        self,
        other: Self | StepExecuting,
    ) -> Self:
        return self.then(other)

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
            A step with fallback behavior. The fallback receives the original input
            state, partial state emitted by the failed step is discarded.

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
            state_emitted: bool = False
            try:
                execution_stream = executing(state=state)
                try:
                    async for chunk in execution_stream:
                        yield chunk

                        if isinstance(chunk, StepState):
                            state_emitted = True

                finally:
                    await execution_stream.aclose()

            except Exception as exc:
                if any(isinstance(exc, exception) for exception in exceptions):
                    ctx.log_info(f"Using fallback Step for {type(exc)}")
                    if state_emitted:
                        yield state  # discard partial state of the failed execution

                    fallback_stream = fallback_executing(state=state)
                    try:
                        async for chunk in fallback_stream:
                            yield chunk

                    finally:
                        await fallback_stream.aclose()

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

        if callable(context):

            async def step(
                state: StepState,
            ) -> StepStream:
                execution_stream = executing(state=state.updating(context=await context()))
                try:
                    async for chunk in execution_stream:
                        if isinstance(chunk, StepState):
                            yield chunk.updating(context=state.context)

                        else:
                            yield chunk

                finally:
                    await execution_stream.aclose()

        else:

            async def step(
                state: StepState,
            ) -> StepStream:
                execution_stream = executing(state=state.updating(context=context))
                try:
                    async for chunk in execution_stream:
                        if isinstance(chunk, StepState):
                            yield chunk.updating(context=state.context)

                        else:
                            yield chunk

                finally:
                    await execution_stream.aclose()

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
            execution_stream = executing(state=state)
            try:
                async for chunk in execution_stream:
                    if isinstance(chunk, StepState):
                        yield chunk.updating(context=state.context)

                    else:
                        yield chunk

            finally:
                await execution_stream.aclose()

        return self.__class__(step)

    def with_volatile_tools(self) -> Self:
        """Discard tool-bearing context elements produced by this step.

        Returns
        -------
        Self
            A wrapped step removing tools from context elements added by the step,
            elements left empty by that are dropped.

        Notes
        -----
        Rationale: keeps transient tool definitions or requests from leaking
        into subsequent step context while preserving other state updates.
        """
        executing: StepExecuting = self._executing

        async def step(
            state: StepState,
        ) -> StepStream:
            execution_stream = executing(state=state)
            try:
                async for chunk in execution_stream:
                    if isinstance(chunk, StepState):
                        common_prefix_length: int = 0
                        for previous, current in zip(
                            state.context,
                            chunk.context,
                            strict=False,
                        ):
                            if previous != current:
                                break

                            common_prefix_length += 1

                        stripped: MutableSequence[ModelContextElement] = []
                        for element in chunk.context[common_prefix_length:]:
                            if not element.contains_tools:
                                stripped.append(element)

                            elif without_tools := element.without_tools():
                                stripped.append(without_tools)

                            # elements consisting only of tools are dropped entirely

                        updated: ModelContext = (
                            *chunk.context[:common_prefix_length],
                            *stripped,
                        )
                        if chunk.context != updated:
                            yield chunk.updating(context=updated)

                        else:
                            yield chunk

                    else:
                        yield chunk

            finally:
                await execution_stream.aclose()

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
                    execution_stream = executing(state=state)
                    try:
                        async for chunk in execution_stream:
                            yield chunk

                    finally:
                        await execution_stream.aclose()

        else:

            async def step(
                state: StepState,
            ) -> StepStream:
                if await condition(state=state):
                    execution_stream = executing(state=state)
                    try:
                        async for chunk in execution_stream:
                            yield chunk

                    finally:
                        await execution_stream.aclose()

                else:
                    alternative_stream = alternative_executing(state=state)
                    try:
                        async for chunk in alternative_stream:
                            yield chunk

                    finally:
                        await alternative_stream.aclose()

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
            execution_stream = executing(state=state)
            try:
                async for chunk in execution_stream:
                    if not isinstance(chunk, StepState):
                        continue  # skip output chunks

                    yield chunk  # pass only state updates

            finally:
                await execution_stream.aclose()

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

            execution_stream = executing(state=state)
            try:
                async for chunk in execution_stream:
                    yield chunk

            finally:
                await execution_stream.aclose()

        return self.__class__(step)

    def with_output_evaluation(
        self,
        evaluator: PreparedEvaluatorScenario[Sequence[StepOutputChunk]]
        | PreparedEvaluator[Sequence[StepOutputChunk]],
        /,
        *,
        raise_on_failure: bool = False,
    ) -> Self:
        """Evaluate all output chunks emitted by this step once it completes.

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
            A wrapped step that evaluates the complete emitted output after the
            wrapped execution ends. Output is already delivered to consumers when
            the evaluation runs, so a failure cannot withhold it.

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
            execution_stream = executing(state=state)
            try:
                async for chunk in execution_stream:
                    yield chunk

                    if isinstance(chunk, StepState):
                        state = chunk  # update local state

                    else:
                        accumulator.append(chunk)

            finally:
                await execution_stream.aclose()

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
            if artifacts or keyed_artifacts:
                raise TypeError("Artifacts can't be provided together with StepState")

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

        try:
            async for chunk in stream:
                if isinstance(chunk, MultimodalContentPart):
                    accumulator.append(chunk)

        finally:
            await stream.aclose()

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
            if artifacts or keyed_artifacts:
                raise TypeError("Artifacts can't be provided together with StepState")

            state = context

        else:
            state = StepState.of(
                context,
                *artifacts,
                **keyed_artifacts,
            )

        stream: StepStream = self._executing(state=state)
        try:
            async for chunk in stream:
                if isinstance(chunk, StepState):
                    state = chunk

        finally:
            await stream.aclose()

        return state

    @overload
    def stream(
        self,
        state: StepState,
        /,
    ) -> AsyncGenerator[StepOutputChunk]: ...

    @overload
    def stream(
        self,
        context: ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> AsyncGenerator[StepOutputChunk]: ...

    async def stream(
        self,
        context: StepState | ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> AsyncGenerator[StepOutputChunk]:
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
        AsyncGenerator[StepOutputChunk]
            Async stream yielding emitted output chunks until completion.

        Raises
        ------
        BaseException
            Propagates exceptions raised by the wrapped step.

        Notes
        -----
        Rationale: exposes streaming execution for incremental consumers such as
        UIs and realtime pipelines.

        Consumers stopping before the stream ends should close it explicitly,
        i.e. through ``ctx.closing``. Leaving it to garbage collection
        finalizes it within an unrelated context, breaking scoped state teardown.
        """

        stream: StepStream
        if isinstance(context, StepState):
            if artifacts or keyed_artifacts:
                raise TypeError("Artifacts can't be provided together with StepState")

            stream = self._executing(state=context)

        else:
            stream = self._executing(
                state=StepState.of(
                    context,
                    *artifacts,
                    **keyed_artifacts,
                ),
            )

        try:
            async for chunk in stream:
                if isinstance(chunk, StepState):
                    continue  # skip state updates

                yield chunk  # provide the rest

        finally:
            await stream.aclose()

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


def _retry_delay(
    delay: Callable[[int, Exception], float] | float | None,
    attempt: int,
    exception: Exception,
    /,
) -> float:
    match delay:
        case None:
            return 0

        case float(strict) | int(strict):
            return float(strict)

        case make_delay:
            return make_delay(attempt, exception)


def _validate_max_iterations(
    max_iterations: int | None,
    /,
) -> None:
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations has to be at least 1")


async def _handling_tool_requests(
    toolbox: Toolbox,
    tool_requests: Sequence[ModelToolRequest],
    /,
    *,
    state: StepState,
) -> StepStream:
    responses: MutableSequence[ModelToolResponse] = []
    tools_output_accumulator: MutableMapping[str, MutableSequence[MultimodalContentPart]] = {}
    tools_stream: AsyncGenerator[ModelToolResponse | ProcessingEvent | MultimodalContentPart] = (
        toolbox.handle(tool_requests)
    )
    try:
        async for chunk in tools_stream:
            if isinstance(chunk, ModelToolResponse):
                responses.append(chunk)

            elif not isinstance(chunk, ProcessingEvent):
                tools_output_accumulator.setdefault(
                    chunk.meta.get_str("request", default=""), []
                ).append(chunk)

            yield chunk

    finally:
        await tools_stream.aclose()

    ctx.log_debug("...received tool responses...")

    if tools_output_accumulator:  # tools direct result, state ends with ModelOutput
        ctx.log_debug("...tools generated output...")
        yield state.appending_context(
            ModelInput.of(*responses),
            ModelOutput.of(
                MultimodalContent.of(
                    *(part for parts in tools_output_accumulator.values() for part in parts)
                )
            ),
        )

    else:  # regular tools result, state ends with ModelInput
        yield state.appending_context(ModelInput.of(*responses))


async def _noop_stream() -> StepStream:
    return  # do not emit anything
    yield  # converts to AsyncGenerator


@Step
async def _noop(
    state: StepState,
) -> StepStream:
    return  # do not emit anything
    yield  # converts to AsyncGenerator


@runtime_checkable
class StepSelecting(Protocol):
    """Protocol for runtime step selection based on the current state."""

    async def __call__(
        self,
        state: StepState,
    ) -> Step: ...
