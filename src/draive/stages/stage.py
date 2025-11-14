from collections.abc import (
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, ClassVar, Literal, Self, cast, final, overload
from uuid import uuid4

from haiway import (
    BasicValue,
    Disposable,
    Disposables,
    Meta,
    MetaValues,
    State,
    cache,
    cache_externally,
    concurrently,
    ctx,
    retry,
)

from draive.evaluation import (
    EvaluatorResult,
    EvaluatorScenarioResult,
    PreparedEvaluator,
    PreparedEvaluatorScenario,
)
from draive.models import (
    FunctionTool,
    GenerativeModel,
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelInstructions,
    ModelMemory,
    ModelMemoryRecall,
    ModelOutput,
    ModelOutputSelection,
    ModelToolRequest,
    ModelToolResponse,
    Tool,
    Toolbox,
)
from draive.multimodal import Multimodal, MultimodalContent, Template, TemplatesRepository
from draive.parameters import DataModel
from draive.stages.types import (
    StageCacheKeyMaking,
    StageCacheReading,
    StageCacheWriting,
    StageConditioning,
    StageContextTransforming,
    StageException,
    StageExecution,
    StageLoopConditioning,
    StageMemory,
    StageMerging,
    StageResultTransforming,
    StageRouting,
    StageState,
)

__all__ = (
    "Stage",
    "stage",
)


@final
class Stage:
    """
    A Stage represents a unit of processing within a model pipeline.

    Stages are immutable execution units that process model context and produce results.
    They can be composed, cached, retried, and conditionally executed to build complex
    AI processing workflows.

    Key Features:
    - Immutable and composable design
    - Built-in caching, retry, and fallback mechanisms
    - Context and result transformation capabilities
    - Support for concurrent and sequential execution
    - Memory integration for state restoration
    - Tool integration for model completion stages
    """

    noop: ClassVar[Self]  # defined after the class
    """
    `noop` stage is a `Stage` which does nothing.
    """

    @overload
    @classmethod
    def predefined(
        cls,
        element: ModelContextElement,
        /,
        *elements: ModelContextElement,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def predefined(
        cls,
        element: Multimodal,
        /,
        *elements: Multimodal,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def predefined(
        cls,
        element: ModelContextElement | Multimodal,
        /,
        *elements: ModelContextElement | Multimodal,
        result: Multimodal | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Insert predefined elements into the context.

        This stage appends the provided elements to the model context. It does not
        modify the current result unless an explicit `result` is provided.
        This is useful for injecting static content (inputs/outputs/tool messages)
        into a processing pipeline.

        Parameters
        ----------
        element : ModelContextElement | Multimodal
            The first element to add to the context.
        *elements : ModelContextElement | Multimodal
            Additional elements to add to the context.
        result : Multimodal | None, default None
            Optional stage result to set explicitly. If None, the current result
            remains unchanged.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A Stage instance that adds predefined elements to the context.

        Examples
        --------
        >>> stage = Stage.predefined(
        ...     "What is 2+2?",
        ...     "The answer is 4."
        ... )
        """

        context: list[ModelContextElement]
        if isinstance(element, ModelContextElement):
            context = [element]

        else:
            context = [ModelInput.of(MultimodalContent.of(element))]

        for value in elements:
            if isinstance(value, ModelContextElement):
                context.append(value)

            elif isinstance(context[-1], ModelInput):
                context.append(ModelOutput.of(MultimodalContent.of(value)))

            else:
                context.append(ModelInput.of(MultimodalContent.of(value)))

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            return state.updated(
                context=(*state.context, *context),
                result=result if result is not None else state.result,
            )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def memory_recall(
        cls,
        memory: StageMemory,
        /,
        *,
        handling: Literal["replace", "merge"] = "replace",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Recall a previously stored stage state from memory.

        This stage retrieves a `StageState` from the provided memory and either
        replaces the current state or merges with it, depending on `handling`.

        Parameters
        ----------
        memory : StageMemory
            Memory from which to recall a `StageState`.
        handling : Literal["replace", "merge"]
            Determines how the recalled state is applied:
            - "replace": Use the recalled state as-is.
            - "merge": Merge the recalled state with the current state.
            Default is "replace".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that recalls context from memory.
        """

        match handling:
            case "replace":

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    async with ctx.scope("stage.memory_recall"):
                        return await memory.recall()

            case "merge":

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    async with ctx.scope("stage.memory_recall"):
                        return state.merged(await memory.recall())

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def memory_remember(
        cls,
        memory: StageMemory,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Store the current stage state in memory.

        This stage saves the current `StageState` to the provided memory instance
        without modifying the context or result.

        Parameters
        ----------
        memory : StageMemory
            Memory in which to store the current `StageState`.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that remembers context in memory.
        """

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.memory_remember"):
                await memory.remember(state)
                return state

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def completion(
        cls,
        input: Template | Multimodal,  # noqa: A002
        /,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Generate a completion for provided input appended to the current context.

        This stage uses the current model context along with the provided input to
        generate a new completion. Both the input and the generated completion are
        added to the context, and the completion becomes the new result value.

        When tools are provided, the underlying model may invoke them; tool calls
        and their results are managed and included in the context.

        Parameters
        ----------
        input : Template | Multimodal
            Input content to provide to the model.
        instructions : Template | ModelInstructions
            Instructions or guidance for the model.
        tools : Toolbox | Iterable[Tool]
            Tools that the model can use during completion generation.
        output : ModelOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the model invocation.

        Returns
        -------
        Self
            A new Stage instance that generates a completion using the model.

        Raises
        ------
        RuntimeError
            If the model exceeds the limit of recursive tool calls.

        Examples
        --------
        >>> stage = Stage.completion(
        ...     "Calculate 25% of 80",
        ...     tools=[calculator]
        ... )
        """

        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.completion"):
                if isinstance(instructions, Template):
                    ctx.record_info(
                        attributes={"instructions.template": instructions.identifier},
                    )

                if isinstance(input, Template):
                    ctx.record_info(
                        attributes={"input.template": input.identifier},
                    )

                context: list[ModelContextElement] = [
                    *state.context,
                    ModelInput.of(await TemplatesRepository.resolve(input)),
                ]
                # Run loop and merge content parts into a single MultimodalContent
                result: ModelOutput = await GenerativeModel.loop(
                    instructions=await TemplatesRepository.resolve_str(instructions),
                    toolbox=toolbox,
                    context=context,
                    output=output,
                    stream=False,
                    **extra,
                )

                return state.updated(
                    context=context,  # GenerativeModel.loop updates the context
                    # Discard reasoning from the stage result; it remains in context
                    result=result.content,
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def prompting_completion(
        cls,
        input: Callable[[], Coroutine[None, None, Multimodal]],  # noqa: A002
        /,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Generate a completion after asynchronously obtaining the input.

        This stage awaits an async function to obtain input content, then uses this
        input along with the current model context to generate a completion.
        The obtained input and the generated completion are added to the context, and
        the completion becomes the new result value.

        As with `completion`, tools may be invoked during generation; tool calls and
        their results are managed automatically.

        Parameters
        ----------
        input : Callable[[], Coroutine[None, None, Multimodal]]
            Async function returning input content to provide to the model.
        instructions : Template | ModelInstructions
            Instructions or guidance for the model.
        tools : Toolbox | Iterable[Tool]
            Tools that the model can use during completion generation.
        output : ModelOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the model invocation.

        Returns
        -------
        Self
            A new Stage instance that generates a completion after prompting for input.

        Raises
        ------
        RuntimeError
            If the model exceeds the limit of recursive tool calls.

        Examples
        --------
        >>> async def get_user_input():
        ...     # Implementation that gets input from user
        ...     return "What is the capital of Poland?"
        ...
        >>> stage = Stage.prompting_completion(
        ...     get_user_input,
        ...     tools=[geography_tool]
        ... )
        """
        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.completion.prompting"):
                ctx.log_debug("Waiting for prompting completion input...")
                context: list[ModelContextElement] = [
                    *state.context,
                    ModelInput.of(MultimodalContent.of(await input())),
                ]
                ctx.log_debug("...prompting completion input provided")
                if isinstance(instructions, Template):
                    ctx.record_info(
                        attributes={"instructions.template": instructions.identifier},
                    )

                result: ModelOutput = await GenerativeModel.loop(
                    instructions=await TemplatesRepository.resolve_str(instructions),
                    toolbox=toolbox,
                    context=context,
                    output=output,
                    stream=False,
                    **extra,
                )

                return state.updated(
                    context=context,  # GenerativeModel.loop updates the context
                    # Discard reasoning from the stage result; it remains in context
                    result=result.content,
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def loopback_completion(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Take the last completion and use it as the new input.

        This stage takes the last ModelOutput completion, replaces the last input with it,
        and executes a new completion using that input. It's useful for iterative refinement
        of outputs where each completion builds upon the previous one.

        Parameters
        ----------
        instructions : Template | ModelInstructions
            Instructions or guidance for the model.
        tools : Toolbox | Iterable[Tool]
            Tools that the model can use during completion generation.
        output : ModelOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the model invocation.

        Returns
        -------
        Self
            A new Stage instance that performs the loopback completion.

        Raises
        ------
        RuntimeError
            If the model exceeds the limit of recursive tool calls.
        """
        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.completion.loopback"):
                if not state.context or not isinstance(state.context[-1], ModelOutput):
                    ctx.log_warning("loopback_completion has been skipped due to invalid context")
                    return state

                # Find the index of the last ModelInput in the context
                last_input_idx: int | None = None
                for idx in range(len(state.context) - 2, -1, -1):
                    if isinstance(state.context[idx], ModelInput):
                        last_input_idx = idx
                        break

                if last_input_idx is None:
                    ctx.log_warning(
                        "loopback_completion could not find a ModelInput in the context"
                    )
                    return state

                # Use last output content as new input
                last_output = state.context[-1]
                assert isinstance(last_output, ModelOutput)  # nosec: B101
                context: list[ModelContextElement] = [
                    *state.context[:last_input_idx],
                    ModelInput.of(last_output.content, meta={"loopback": True}),
                ]

                if isinstance(instructions, Template):
                    ctx.record_info(
                        attributes={"instructions.template": instructions.identifier},
                    )

                result: ModelOutput = await GenerativeModel.loop(
                    instructions=await TemplatesRepository.resolve_str(instructions),
                    toolbox=toolbox,
                    context=context,
                    output=output,
                    stream=False,
                    **extra,
                )

                return state.updated(
                    context=context,  # GenerativeModel.loop updates the context
                    # Discard reasoning from the stage result; it remains in context
                    result=result.content,
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def result_completion(
        cls,
        *,
        instructions: Template | ModelInstructions = "",
        tools: Toolbox | Iterable[Tool] = (),
        output: ModelOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Use the current result as the next input and generate a completion.

        This stage takes the current stage result, appends it as a new input, and executes
        a new completion using the updated context.

        Parameters
        ----------
        instructions : Template | ModelInstructions
            Instructions or guidance for the model.
        tools : Toolbox | Iterable[Tool]
            Tools that the model can use during completion generation.
        output : ModelOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the model invocation.

        Returns
        -------
        Self
            A new Stage instance that performs the result completion.

        Raises
        ------
        RuntimeError
            If the model exceeds the limit of recursive tool calls.
        """
        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.completion.result"):
                if isinstance(instructions, Template):
                    ctx.record_info(
                        attributes={"instructions.template": instructions.identifier},
                    )

                context: list[ModelContextElement] = [
                    *state.context,
                    ModelInput.of(state.result),
                ]
                result: ModelOutput = await GenerativeModel.loop(
                    instructions=await TemplatesRepository.resolve_str(instructions),
                    toolbox=toolbox,
                    context=context,
                    output=output,
                    stream=False,
                    **extra,
                )

                return state.updated(
                    context=context,  # GenerativeModel.loop updates the context
                    # Discard reasoning from the stage result; it remains in context
                    result=result.content,
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def transform_result(
        cls,
        transformation: StageResultTransforming,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Transform the current result without altering the context.

        This stage applies a transformation function to the current result to produce a new
        result value, while keeping the `ModelContext` unchanged.

        Parameters
        ----------
        transformation : StageResultTransforming
            Function that takes the current result and returns a transformed result.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that transforms the result.
        """

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.transform.result"):
                return state.updated(result=await transformation(state.result))

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def transform_context(
        cls,
        transformation: StageContextTransforming,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Transform the current context without altering the result.

        This stage applies a transformation function to the current `ModelContext` to produce a
        new context, while keeping the result unchanged.

        Parameters
        ----------
        transformation : StageContextTransforming
            Function that takes the current context and returns a transformed context.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that transforms the context.
        """

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.transform.context"):
                return state.updated(context=await transformation(state.context))

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def trim_context(
        cls,
        *,
        limit: slice | int | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Trim the current context to a specified limit.

        This stage reduces the size of the `ModelContext` by applying a slice or index limit,
        or by clearing it entirely if no limit is specified.

        Parameters
        ----------
        limit : slice | int | None
            Specifies how to trim the context:
            - None: Clear the context completely
            - int: Keep only the first N elements
            - slice: Apply a slice operation to the context
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that trims the context.
        """
        match limit:
            case None:

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    return state.updated(context=())

            case int() as index:

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    return state.updated(context=state.context[:index])

            case slice() as index_slice:

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    return state.updated(context=state.context[index_slice])

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def strip_context_tools(
        cls,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Trim the current context by removing tool calls.

        Parameters
        ----------
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that removes tool-related elements from the context.
        """

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            return state.updated(
                context=tuple(element.without_tools() for element in state.context)
            )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def tool_call[**Args, Result](
        cls,
        _tool: FunctionTool[Args, Result],
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Self:
        """
        Execute a tool and add its result to the context.

        This stage calls the provided tool with the given arguments and adds
        the tool call to the model context as proper tool request/response pairs.
        This makes the tool interaction visible in the context for subsequent stages.

        Warning:
        --------
        Models usually require tool calls to be between regular input/completion messages.
        You may need to manually adjust context afterwards to ensure proper contents.

        Parameters
        ----------
        _tool : FunctionTool[Args, Result]
            The tool to execute.
        *args : Args.args
            Positional arguments to pass to the tool.
        **kwargs : Args.kwargs
            Keyword arguments to pass to the tool.

        Returns
        -------
        Self
            A new Stage instance that executes the tool and adds its result to context.

        Examples
        --------
        >>> @tool
        ... async def get_weather(location: str) -> str:
        ...     return f"Weather in {location}: sunny"
        ...
        >>> stage = Stage.tool_call(get_weather, location="New York")
        """
        assert not args, "Positional arguments are not supported"  # nosec: B101

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.tool_call"):
                request_id: str = uuid4().hex
                # Create tool request representing the call
                tool_request: ModelToolRequest = ModelToolRequest.of(
                    request_id,
                    tool=_tool.name,
                    arguments=kwargs,
                )

                try:
                    tool_response: ModelToolResponse
                    result: MultimodalContent
                    if _tool.handling == "detached":
                        ctx.spawn(_tool.call, request_id, **kwargs)
                        tool_response = ModelToolResponse.of(
                            tool_request.identifier,
                            tool=_tool.name,
                            content=MultimodalContent.of(
                                f"{_tool.name} tool execution has been requested"
                            ),
                            handling="detached",
                        )
                        result = state.result

                    else:
                        tool_result: MultimodalContent = await _tool.call(request_id, **kwargs)
                        match _tool.handling:
                            case "response":
                                tool_response = ModelToolResponse.of(
                                    tool_request.identifier,
                                    tool=_tool.name,
                                    content=tool_result,
                                    handling="response",
                                )
                                result = state.result

                            case "output":
                                tool_response = ModelToolResponse.of(
                                    tool_request.identifier,
                                    tool=_tool.name,
                                    content=tool_result,
                                    handling="output",
                                )
                                result = tool_result

                            case "output_extension":
                                tool_response = ModelToolResponse.of(
                                    tool_request.identifier,
                                    tool=_tool.name,
                                    content=tool_result,
                                    handling="output_extension",
                                )
                                result = state.result.appending(tool_result)

                            case _:
                                tool_response = ModelToolResponse.of(
                                    tool_request.identifier,
                                    tool=_tool.name,
                                    content=tool_result,
                                    handling="response",
                                )
                                result = state.result

                except Exception as exc:
                    tool_response = ModelToolResponse.of(
                        tool_request.identifier,
                        tool=_tool.name,
                        content=MultimodalContent.of(_tool.format_error(exc)),
                        handling="error",
                    )
                    result = state.result

                return state.updated(
                    context=(
                        *state.context,
                        ModelOutput.of(tool_request),
                        ModelInput.of(tool_response),
                    ),
                    result=result,
                )

        return cls(
            stage,
            meta=Meta.of({"tool_call": _tool.name}),
        )

    @classmethod
    def result_evaluation(
        cls,
        evaluator: PreparedEvaluatorScenario[MultimodalContent]
        | PreparedEvaluator[MultimodalContent],
        /,
        *,
        raises: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Evaluate the current result using an evaluator.

        This stage takes the current stage result and runs it through the provided
        evaluator or scenario evaluator. The stage raises StageException when evaluation fails.

        Parameters
        ----------
        evaluator : PreparedEvaluatorScenario[MultimodalContent]
        | PreparedEvaluator[MultimodalContent]
            The evaluator or scenario evaluator to use for evaluation.
        raises: bool = False
            Determines whether to raise ``StageException`` when the evaluation fails.
            When ``False``, the stage returns the input state unchanged on failure.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that evaluates the result.

        Examples
        --------
        >>> stage = Stage.result_evaluation(evaluator)
        """

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.result_evaluation"):
                evaluation_result: EvaluatorScenarioResult | EvaluatorResult = await evaluator(
                    state.result
                )

                if evaluation_result.passed or not raises:
                    return state  # evaluation passed, keep going

                performance: float = evaluation_result.performance
                report: str = evaluation_result.report(detailed=__debug__)
                raise StageException(
                    f"Result evaluation failed with relative score: {performance:.2f}%",
                    state=state,
                    meta={
                        "evaluation_performance": performance,
                        "evaluation_report": report,
                    },
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def context_evaluation(
        cls,
        evaluator: PreparedEvaluatorScenario[ModelContext] | PreparedEvaluator[ModelContext],
        /,
        *,
        raises: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Evaluate the current context using an evaluator.

        This stage takes the current model context and runs it through the provided
        evaluator or scenario evaluator. The stage raises StageException when evaluation fails.

        Parameters
        ----------
        evaluator : PreparedEvaluatorScenario[Value] | PreparedEvaluator[Value]
            The evaluator or scenario evaluator to use for evaluation.
        raises: bool = False
            Determines whether to raise ``StageException`` when the evaluation fails.
            When ``False``, the stage returns the input state unchanged on failure.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that evaluates the context.

        Examples
        --------
        >>> stage = Stage.context_evaluation(evaluator)
        """

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.context_evaluation"):
                evaluation_result: EvaluatorScenarioResult | EvaluatorResult = await evaluator(
                    state.context
                )

                if evaluation_result.passed or not raises:
                    return state  # evaluation passed, keep going

                performance: float = evaluation_result.performance
                report: str = evaluation_result.report(detailed=__debug__)
                raise StageException(
                    f"Context evaluation failed with relative score: {performance:.2f}%",
                    state=state,
                    meta={
                        "evaluation_performance": performance,
                        "evaluation_report": report,
                    },
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def loop(
        cls,
        stage: Self,
        /,
        *stages: Self,
        condition: StageLoopConditioning,
        condition_check: Literal[
            "before_execution",
            "after_execution",
        ] = "after_execution",
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Execute one or more stages in a loop while a condition is met.

        This Stage repeatedly executes the provided Stage as long as the condition
        function returns True when applied to the current context and result.

        Parameters
        ----------
        stage : Stage
            The Stage to execute repeatedly.
        condition : StageLoopConditioning
            A function that determines whether to continue looping. It should be
            an async function that accepts the current state and iteration number
            and returns a boolean.
        condition_check: Literal["before_execution", "after_execution"] = "after_execution"
            A loop mode determining the first condition check behavior.
            "before_execution" will check the condition before executing stage
            "after_execution" will check the condition after executing stage
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Stage
            A new Stage instance that implements the looping behavior.
        """
        stage_executions: Sequence[StageExecution] = (
            stage,
            *(stage for stage in stages),
        )

        match condition_check:
            case "before_execution":

                async def stage_loop(
                    *,
                    state: StageState,
                ) -> StageState:
                    async with ctx.scope("stage.loop"):
                        current_state: StageState = state
                        iteration: int = 0

                        while await condition(state=current_state, iteration=iteration):
                            async with ctx.scope("stage.loop.iteration"):
                                for execution in stage_executions:
                                    current_state = await execution(state=current_state)

                                iteration += 1

                        return current_state

            case "after_execution":

                async def stage_loop(
                    *,
                    state: StageState,
                ) -> StageState:
                    async with ctx.scope("stage.loop"):
                        current_state: StageState = state
                        iteration: int = 0

                        while True:
                            async with ctx.scope("stage.loop.iteration"):
                                for stage_execution in stage_executions:
                                    current_state = await stage_execution(state=current_state)

                                if not await condition(state=current_state, iteration=iteration):
                                    return current_state

                                iteration += 1

        return cls(
            stage_loop,
            meta=Meta.of(meta),
        )

    @classmethod
    def sequence(
        cls,
        stage: Self,
        /,
        *stages: Self,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Execute multiple stages in sequence.

        This Stage chains the execution of the provided Stages, where each Stage
        receives the context and result produced by the previous Stage.

        Parameters
        ----------
        stage : Stage
            The first Stage in the sequence.
        *stages : Stage
            Additional Stages to execute in order after the first Stage.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Stage
            A new Stage instance that executes the provided Stages sequentially.
        """
        stage_executions: Sequence[StageExecution] = tuple(stage for stage in (stage, *stages))

        async def stage_sequence(
            *,
            state: StageState,
        ) -> StageState:
            current_state: StageState = state

            for execution in stage_executions:
                current_state = await execution(state=current_state)

            return current_state

        return cls(
            stage_sequence,
            meta=Meta.of(meta),
        )

    @classmethod
    def concurrent(
        cls,
        stage: Self,
        /,
        *stages: Self,
        merge: StageMerging,
        concurrent_tasks: int = 2,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Execute multiple stages concurrently and merge their results.

        This Stage runs all provided Stages in parallel using the same initial
        context and result, then combines their results using the specified
        merge function.

        Parameters
        ----------
        stage : Stage
            The first Stage to execute concurrently.
        *stages : Stage
            Additional Stages to execute concurrently with the first Stage.
        merge : StageMerging
            A function that merges the results from all concurrent Stage executions.
            It should be an async function that accepts a sequence of (context, result)
            tuples and returns a single (context, result) tuple.
        concurrent_tasks: int = 2
            Number of concurrently running tasks.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Stage
            A new Stage instance that executes Stages concurrently and merges results.
        """
        stage_executions: Sequence[StageExecution] = tuple(stage for stage in (stage, *stages))

        async def concurrent_stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.concurrent"):
                return await merge(
                    branches=cast(  # we are converting errors witin __call__
                        Sequence[StageState | StageException],
                        await concurrently(
                            (execution(state=state) for execution in stage_executions),
                            concurrent_tasks=concurrent_tasks,
                            return_exceptions=True,
                        ),
                    )
                )

        return cls(
            concurrent_stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def router(
        cls,
        stage: Self,
        /,
        *stages: Self,
        routing: StageRouting | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Route execution to one of several stages based on a routing function.

        This Stage uses a routing function to select which of the provided stages to execute
        based on the current context and result. Each stage is identified by its metadata name
        (required) and description (required). Stage names are normalized to lowercase for matching.

        By default, uses LLM-based routing that analyzes the context and result to intelligently
        select the most appropriate stage based on stage descriptions.

        Parameters
        ----------
        stage : Stage
            The first stage option for routing. Must have name and description in metadata.
        *stages : Stage
            Additional stage options for routing. Each must have name and description in metadata.
        routing : StageRouting, optional
            A function that selects which stage to execute. Defaults to `_lmm_routing` which
            uses LLM analysis. The function receives the current context, result, and available
            options, and returns the key of the selected stage.
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.

        Returns
        -------
        Self
            A new Stage instance that routes execution to the selected stage.

        Raises
        ------
        AssertionError
            If any stage is missing required name or description metadata.
        ValueError
            If the routing function returns a selection that doesn't match any available stage.

        Examples
        --------
        >>> # Using default LLM-based routing
        >>> router = Stage.router(
        ...     normal_stage.with_meta(
        ...         name="process_data",
        ...         description="Process and analyze the input data"
        ...     ),
        ...     error_stage.with_meta(
        ...         name="handle_error",
        ...         description="Handle errors and provide error responses"
        ...     )
        ... )
        >>>
        >>> # Using custom routing function
        >>> async def custom_routing(
        ...     context: LMMContext,
        ...     result: MultimodalContent,
        ...     options: Mapping[str, Meta],
        ... ):
        ...     if "error" in result.as_string():
        ...         return "handle_error"
        ...     return "process_data"
        ...
        >>> router = Stage.router(
        ...     normal_stage.with_meta(name="process_data", description="..."),
        ...     error_stage.with_meta(name="handle_error", description="..."),
        ...     routing=custom_routing
        ... )
        """
        if not stages:  # there is no selection when there is only one option
            ctx.log_debug("Stage.router prepared with single option, skipping routing...")
            return stage

        routes: MutableMapping[str, StageExecution] = {}
        options: MutableMapping[str, Meta] = {}
        for route in (stage, *stages):
            assert route.meta.name, "Stage names are required for routing"  # nosec: B101
            assert route.meta.description, "Stage descriptions are required for routing"  # nosec: B101
            key: str = (route.meta.name or uuid4().hex).strip().lower()
            assert "\n" not in key, "Stage names can't have newlines"  # nosec: B101
            routes[key] = route._execution
            options[key] = route.meta

        routing = routing if routing is not None else _model_routing

        async def router_stage(
            *,
            state: StageState,
        ) -> StageState:
            async with ctx.scope("stage.router"):
                selection: str = await routing(
                    state=state,
                    options=options,
                )

            # out of scope - executing selected
            if selected := routes.get(selection):
                return await selected(state=state)

            else:
                raise RuntimeError(
                    f"Stage.router selection invalid - selection ({selection}) is not available"
                )

        return cls(
            router_stage,
            meta=Meta.of(meta),
        )

    __slots__ = (
        "_execution",
        "meta",
    )

    def __init__(
        self,
        execution: StageExecution,
        /,
        meta: Meta,
    ) -> None:
        assert not isinstance(execution, Stage)  # nosec: B101
        assert isinstance(execution, StageExecution)  # nosec: B101
        self._execution: StageExecution
        object.__setattr__(
            self,
            "_execution",
            execution,
        )
        self.meta: Meta
        object.__setattr__(
            self,
            "meta",
            meta,
        )

    def with_meta(
        self,
        **values: BasicValue,
    ) -> Self:
        """
        Update metadata for this stage.

        This method allows you to add or update metadata fields for the Stage,
        such as name, description, tags, and other custom metadata values.
        The metadata is particularly important for routing stages where name
        and description are required.

        Parameters
        ----------
        **values : BasicValue
            Metadata values to add or update. Common values include:
            - name: str - A unique identifier for the stage
            - description: str - A description of what the stage does
            - tags: Sequence[str] - Tags for categorizing the stage
            - Any other custom metadata fields

        Returns
        -------
        Self
            A new Stage instance with the updated metadata.

        Examples
        --------
        >>> stage = Stage.completion("Calculate 2+2")
        >>> named_stage = stage.with_meta(
        ...     name="calculator",
        ...     description="Performs basic arithmetic calculations",
        ...     tags=["math", "arithmetic"]
        ... )
        >>>
        >>> # Required for routing stages
        >>> router = Stage.router(
        ...     stage1.with_meta(name="option1", description="First option"),
        ...     stage2.with_meta(name="option2", description="Second option"),
        ...     routing=my_routing_function
        ... )
        """
        return self.__class__(
            self._execution,
            meta=self.meta.merged_with(values),
        )

    @overload
    def with_ctx(
        self,
        /,
        *,
        disposables: Disposables | Collection[Disposable],
    ) -> Self: ...

    @overload
    def with_ctx(
        self,
        state_context: State,
        /,
        *state: State,
    ) -> Self: ...

    def with_ctx(  # noqa: C901
        self,
        state: State | None = None,
        /,
        *states: State,
        disposables: Disposables | Collection[Disposable] | None = None,
    ) -> Self:
        """
        Apply the specified state context to this stage.

        The returned Stage will execute within the provided state context,
        which affects what contextual state is available.

        Note that Processing state will be replaced with contextual.

        Parameters
        ----------
        state_context : State | None
            `State` object to include in the state context update.

        *state : State
            Optional additional `State` objects to include in the state context.
            Only applicable when `state_context` is a `State`.

        disposables: Disposables | Collection[Disposable] | None
            Optional Disposables which will be used for execution of this stage.
            State produced by disposables will be used within the context state.

        Returns
        -------
        Stage
            A new Stage instance that executes within the specified context.
        """
        execution: StageExecution = self._execution

        resolved_disposables: Disposables | None
        if disposables is None:
            resolved_disposables = None

        elif isinstance(disposables, Disposables):
            resolved_disposables = disposables

        else:
            resolved_disposables = Disposables(disposables)

        match (state, resolved_disposables):
            case (None, None):
                assert not states  # nosec: B101
                return self  # nothing to change...

            case (None, ctx_disposables):
                assert not states  # nosec: B101

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    result_state: StageState
                    try:
                        with ctx.updated(*await ctx_disposables.prepare()):
                            result_state = await execution(state=state)

                    except BaseException as exc:
                        await ctx_disposables.dispose(
                            exc_type=type(exc),
                            exc_val=exc,
                            exc_tb=exc.__traceback__,
                        )
                        raise exc

                    else:
                        await ctx_disposables.dispose()

                    return result_state

            case (ctx_state, None):

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    with ctx.updated(ctx_state, *states):
                        return await execution(state=state)

            case (ctx_state, ctx_disposables):

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    disposables_state: Iterable[State] = await ctx_disposables.prepare()
                    result_state: StageState
                    try:
                        with ctx.updated(ctx_state, *disposables_state, *states):
                            result_state = await execution(state=state)

                    except BaseException as exc:
                        await ctx_disposables.dispose(
                            exc_type=type(exc),
                            exc_val=exc,
                            exc_tb=exc.__traceback__,
                        )
                        raise exc

                    else:
                        await ctx_disposables.dispose()

                    return result_state

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def cached(
        self,
        *,
        limit: int | None = None,
        expiration: float | None = None,
    ) -> Self:
        """
        Enable caching for this stage.

        The returned Stage will provide the same context and result as the last execution
        if cache keys match.
        Note that the result will be the exact same context and result each time regarless
        of cached stage actual execution assuming no side effects or processing state changes
        are produced within cached stage.

        Parameters
        ----------
        limit : int | None
            Maximum number of entries to store in the cache. If None, cache size is
            one.
        expiration : float | None
            Time in seconds after which cache entries expire. If None, entries never
            expire.

        Returns
        -------
        Self
            A new Stage instance with caching enabled.
        """

        return self.__class__(
            cache(
                limit=limit,
                expiration=expiration,
            )(self._execution),
            meta=self.meta,
        )

    def cached_externally[Key](
        self,
        *,
        make_key: StageCacheKeyMaking[Key],
        read: StageCacheReading[Key],
        write: StageCacheWriting[Key],
    ) -> Self:
        """
        Enable external caching for this stage.

        The returned Stage will provide the same context and result as the last execution
        if cache keys match.
        Note that the result will be the exact same context and result each time regarless
        of cached stage actual execution assuming no side effects or processing state changes
        are produced within cached stage.

        Parameters
        ----------
        make_key : StageCacheKeyMaking[Key]
            Custom function to generate cache keys from execution inputs. The function
            should accept context and result parameters and return a hashable value
            uniquely identifying the inputs.
        read : StageCacheReading[Key]
            Custom function to retrieve cached values from external source.
        write : StageCacheWriting[Key]
            Custom function to store values in external cache.

        Returns
        -------
        Self
            A new Stage instance with caching enabled.
        """

        return self.__class__(
            cache_externally(
                make_key=make_key,
                read=read,
                write=write,
            )(self._execution),
            meta=self.meta,
        )

    def with_retry(
        self,
        *,
        limit: int = 1,
        delay: Callable[[int, Exception], float] | float | None = None,
        catching: Callable[[Exception], bool] | type[Exception] = Exception,
    ) -> Self:
        """
        Add retry behavior to this stage.

        The returned Stage will re-execute its processing function if it raises
        any of the exceptions specified in `catching`, up to `limit` times.
        Retries can be delayed using a fixed `delay` in seconds or a dynamic
        `delay` function that calculates the delay based on the retry attempt and exception.

        Parameters
        ----------
        limit : int
            Maximum number of retry attempts. Defaults to 1.
        delay : Callable[[int, Exception], float] | float | None
            Delay between retries in seconds. Can be a fixed float, a function
            that accepts the retry attempt number (starting from 0) and the raised
            exception and returns the delay, or None for no delay. Defaults to None.
        catching : Callable[[Exception], bool] | type[Exception]
            Predicate or exception type that determine whether the raised
            exception should trigger a retry. Defaults to catching all Exceptions.

        Returns
        -------
        Self
            A new Stage instance with retry behavior.
        """

        return self.__class__(
            retry(
                limit=limit,
                delay=delay,
                catching=catching,
            )(self._execution),
            meta=self.meta,
        )

    def with_fallback(
        self,
        fallback: Self,
        *,
        catching: set[type[Exception]] | tuple[type[Exception], ...] | type[Exception] = Exception,
    ) -> Self:
        """
        Add fallback behavior to this stage.

        The returned Stage will execute fallback stage if it raises
        any of the exceptions specified in `catching`.

        Parameters
        ----------
        fallback : Stage
            Stage to be executed as fallback.
        catching : set[type[Exception]] | tuple[type[Exception], ...] | type[Exception]
            Exception types to catch and fallback on. Defaults to catching all Exceptions.

        Returns
        -------
        Self
            A new Stage instance with fallback behavior.
        """

        execution: StageExecution = self._execution
        fallback_execution: StageExecution = fallback._execution
        exceptions: Collection[type[Exception]] = (
            catching if isinstance(catching, set | tuple) else {catching}
        )

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            try:
                return await execution(state=state)

            except Exception as exc:
                if any(isinstance(exc, exception) for exception in exceptions):
                    ctx.log_info(f"Using fallback stage for {type(exc)}")
                    return await fallback_execution(state=state)

                raise exc

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def with_volatile_context(self) -> Self:
        """
        Discard any context changes produced by this stage.

        The returned stage executes normally but reverts any changes made to the
        model context when completed, keeping only the result changes.

        Returns
        -------
        Stage
            A new stage instance that discards context changes.
        """
        execution: StageExecution = self._execution

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            processed_state: StageState = await execution(state=state)
            return processed_state.updated(context=state.context)

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def with_volatile_tools_context(self) -> Self:
        """
        Discard any tool calls added by this stage.

        The returned stage executes normally, but upon completion, it removes any
        tool request/response blocks added to the context during this stage's
        execution. If previously existing tool usage is part of the common prefix
        with the prior context, it is preserved.

        Returns
        -------
        Stage
            A new stage instance that makes tool-related context elements volatile.
        """
        execution: StageExecution = self._execution

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            processed_state: StageState = await execution(state=state)

            common_prefix: ModelContext = tuple(
                current
                for previous, current in zip(
                    state.context,
                    processed_state.context,
                    strict=False,
                )
                if current == previous
            )
            suffix: ModelContext = processed_state.context[len(common_prefix) :]

            return state.updated(
                context=(
                    *common_prefix,
                    *(element.without_tools() for element in suffix),
                ),
                result=processed_state.result,
            )

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def when(
        self,
        condition: StageConditioning | bool,
        /,
        *,
        alternative: Self | None = None,
    ) -> Self:
        """
        Conditionally execute this stage based on a predicate.

        This returns a new stage that executes the current stage only if the
        specified condition is met. If the condition is not met and an
        alternative stage is provided, that alternative is executed instead.

        Parameters
        ----------
        condition : StageConditioning | bool
            A boolean value or async predicate that determines whether this stage executes.
        alternative : Stage | None
            Optional stage to execute when the condition is not met.

        Returns
        -------
        Stage
            A new stage instance that applies conditional execution logic.

        Examples
        --------
        >>> # Execute stage only when result contains specific text
        >>> async def has_error(context, result):
        ...     return "error" in result.to_str()
        >>>
        >>> error_handling = my_stage.when(has_error)
        >>>
        >>> # Use a boolean condition with an alternative
        >>> primary_stage.when(is_premium_user, alternative=basic_stage)
        """
        execution: StageExecution = self._execution
        alternative_execution: StageExecution | None = (
            alternative._execution if alternative else None
        )

        match condition:
            case bool() as value:

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    if value:
                        return await execution(state=state)

                    elif alternative_execution:
                        return await alternative_execution(state=state)

                    else:
                        return state

            case function:

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    if await function(state=state):
                        return await execution(state=state)

                    elif alternative_execution:
                        return await alternative_execution(state=state)

                    else:
                        return state

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def ignore_result(self) -> Self:
        """
        Ignore the result produced by this stage and preserve the incoming result.

        The returned stage executes normally but retains the original result
        value rather than replacing it with the produced result.

        Returns
        -------
        Stage
            A new Stage instance that ignores its produced result.
        """
        execution: StageExecution = self._execution

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            processed_state: StageState = await execution(state=state)
            return processed_state.updated(result=state.result)  # preserve current result

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def extend_result(self) -> Self:
        """
        Extend the current result by appending the result produced by this stage.

        The returned stage executes normally but appends its result to the
        current result rather than replacing it.

        Returns
        -------
        Stage
            A new stage instance that extends rather than replaces the result.
        """
        execution: StageExecution = self._execution

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            processed_state: StageState = await execution(state=state)
            return processed_state.updated(
                result=state.result.appending(processed_state.result),
            )

        return self.__class__(
            stage,
            meta=self.meta,
        )

    def __setattr__(
        self,
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError("Stage is frozen and can't be modified")

    def __delattr__(
        self,
        __name: str,
    ) -> None:
        raise RuntimeError("Stage is frozen and can't be modified")

    async def __call__(
        self,
        *,
        state: StageState,
    ) -> StageState:
        try:
            return await self._execution(state=state)

        except StageException as exc:
            raise exc

        except Exception as exc:
            raise StageException(
                f"Stage execution failed: {exc}",
                state=state,
            ) from exc

    async def execute(
        self,
        *state: DataModel | State,
        memory: ModelMemory | ModelContext = (),
        result: MultimodalContent = MultimodalContent.empty,
    ) -> MultimodalContent:
        """
        Execute this stage with the provided initial state/context and return the result.

        This method runs the stage's processing function with the specified
        initial context (provided directly or recalled from `memory`), result,
        and additional state, then returns the produced result. When a
        `Memory` is provided, any newly produced context elements will be
        remembered after execution.

        Parameters
        ----------
        *state : DataModel | State
            Initial state models to use for execution.
        memory : ModelMemory | ModelContext
            Either a concrete `ModelContext` to start with, or a `ModelMemory`
            from which the initial context will be recalled and to which any
            newly produced context will be persisted.
        result : MultimodalContent
            Initial result value to use (defaults to empty).

        Returns
        -------
        MultimodalContent
            The result produced by executing the stage.
        """
        async with ctx.scope("stage.execution"):
            initial_context: ModelContext
            if isinstance(memory, ModelMemory):
                recalled: ModelMemoryRecall = await memory.recall()
                initial_context = recalled.context

            else:
                initial_context = memory

            result_state: StageState = await self(
                state=StageState.of(
                    *state,
                    context=initial_context,
                    result=result,
                )
            )

            if isinstance(memory, ModelMemory):
                # Persist only newly produced context elements to the model memory
                # (compute the suffix after the common prefix with the recalled context)
                common_prefix: ModelContext = tuple(
                    current
                    for previous, current in zip(
                        initial_context,
                        result_state.context,
                        strict=False,
                    )
                    if current == previous
                )
                suffix: ModelContext = result_state.context[len(common_prefix) :]
                if suffix:
                    await memory.remember(*suffix)

            return result_state.result


async def _model_routing(
    *,
    state: StageState,
    options: Mapping[str, Meta],
) -> str:
    options_text = "\n".join(
        [f"- {key}: {meta.description or 'No description'}" for key, meta in options.items()]
    )

    instructions: str = (
        "Based on the provided context and the current result,"  # nosec: B608 - false positive
        " select the most appropriate option from the following:"
        f"\n\n{options_text}"
        "\n\nRespond with with the exact option name within SELECTION xml tag"
        f" like (e.g., '<SELECTION>{next(iter(options.keys()))}</SELECTION>'"
    )

    # Create routing context with the current result as input
    routing_context: list[ModelContextElement] = [
        *state.context,
        ModelInput.of(
            MultimodalContent.of(
                "<RESULT>",
                state.result,
                "</RESULT>",
            )
        ),
    ]

    result: ModelOutput = await GenerativeModel.loop(
        instructions=instructions,
        context=routing_context,
        output="text",
    )

    selection_tag = result.content.tag("SELECTION")
    selection: str = (
        selection_tag.content.to_str().strip().lower()
        if selection_tag is not None
        else "__INVALID_SELECTION__"
    )

    if selection not in options:
        raise ValueError(
            f"Model routing failed: invalid selection '{selection}'. "
            f"Available options: {list(options.keys())}"
        )

    return selection


@overload
def stage(
    execution: StageExecution,
    /,
) -> Stage: ...


@overload
def stage(
    *,
    meta: Meta | MetaValues | None = None,
) -> Callable[[StageExecution], Stage]: ...


def stage(
    execution: StageExecution | None = None,
    /,
    *,
    meta: Meta | MetaValues | None = None,
) -> Callable[[StageExecution], Stage] | Stage:
    """Convert an async execution function into a ``Stage``.

    Can be used as a decorator or called directly. When used as a decorator without
    arguments, it wraps the function into a ``Stage`` with default metadata. When
    called, it returns a decorator that applies the provided metadata.

    Parameters
    ----------
    execution : StageExecution | None
        Execution function to wrap. When ``None``, returns a decorator.
    meta : Meta | Mapping[str, Any] | None, optional
        Metadata to attach to the created stage (e.g., name, description, tags).

    Returns
    -------
    Callable[[StageExecution], Stage] | Stage
        A decorator when ``execution`` is ``None``; otherwise, the constructed ``Stage``.
    """

    def wrap(execution: StageExecution) -> Stage:
        return Stage(execution, meta=Meta.of(meta))

    if execution is None:
        return wrap

    else:
        return wrap(execution)


@stage
async def _noop(
    *,
    state: StageState,
) -> StageState:
    return state


Stage.noop = _noop
