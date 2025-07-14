from asyncio import gather
from collections.abc import (
    Callable,
    Collection,
    Coroutine,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, ClassVar, Literal, Protocol, Self, cast, final, overload
from uuid import uuid4

from haiway import Disposable, Disposables, State, cache, ctx, retry

from draive.commons import Meta, MetaValue, MetaValues
from draive.evaluation import (
    EvaluatorResult,
    PreparedEvaluator,
    PreparedScenarioEvaluator,
    ScenarioEvaluatorResult,
)
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContext,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponse,
    LMMToolResponses,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel
from draive.prompts import Prompt
from draive.stages.types import (
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
from draive.tools import FunctionTool, Tool, Toolbox

__all__ = (
    "Stage",
    "stage",
)


class MakeCacheKey[Key](Protocol):
    """Protocol for generating cache keys from stage state."""

    def __call__(
        self,
        *,
        state: StageState,
    ) -> Key: ...


class ReadCache[Key](Protocol):
    """Protocol for reading cached stage states."""

    async def __call__(
        self,
        key: Key,
    ) -> StageState | None: ...


class WriteCache[Key](Protocol):
    """Protocol for writing stage states to cache."""

    async def __call__(
        self,
        key: Key,
        value: StageState,
    ) -> None: ...


@final
class Stage:
    """
    A Stage represents a unit of processing within an LMM pipeline.

    Stages are immutable execution units that process LMM context and produce results.
    They can be composed, cached, retried, and conditionally executed to build complex
    AI processing workflows.

    Key Features:
    - Immutable and composable design
    - Built-in caching, retry, and fallback mechanisms
    - Context and result transformation capabilities
    - Support for concurrent and sequential execution
    - Memory integration for conversation state
    - Tool integration for LMM completion stages
    """

    noop: ClassVar[Self]  # defined after the class
    """
    `noop` stage is a `Stage` which does nothing.
    """

    @overload
    @classmethod
    def predefined(
        cls,
        element: LMMContextElement,
        /,
        *elements: LMMContextElement,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def predefined(
        cls,
        element: Prompt | Multimodal,
        /,
        *elements: Multimodal,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def predefined(
        cls,
        element: Prompt | LMMContextElement | Multimodal,
        /,
        *elements: LMMContextElement | Multimodal,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Creates a Stage that inserts predefined elements into the context.

        This Stage adds the specified elements to the LMM context and
        sets the result to the last provided completion value. It's useful for
        injecting static content into a processing pipeline.
        Content of a last LMMCompletion in the context will be used as stage result.
        Note that context should always end with LMMCompletion as the result.

        Parameters
        ----------
        element : Prompt | LMMContextElement | Multimodal
            The first element to add to the context.
        *elements : LMMContextElement | Multimodal
            Additional elements to add to the context.
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

        context: list[LMMContextElement]
        match element:
            case context_element if isinstance(context_element, LMMContextElement):
                context = [context_element]

            case Prompt() as prompt:
                context = [*prompt.content]

            case content:
                context = [LMMInput.of(MultimodalContent.of(content))]

        for idx, value in enumerate(elements):
            if isinstance(value, LMMContextElement):
                context.append(value)

            elif idx % 2 == 0:
                context.append(LMMInput.of(MultimodalContent.of(value)))

            else:
                context.append(LMMCompletion.of(MultimodalContent.of(value)))

        context_extension: LMMContext = tuple(context)
        completion_result: MultimodalContent | None = next(
            (
                item.content
                for item in reversed(context_extension)
                if isinstance(item, LMMCompletion)
            ),
            None,
        )

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            with ctx.scope("stage.predefined"):
                return state.updated(
                    context=(*state.context, *context_extension),
                    result=completion_result if completion_result is not None else state.result,
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
        Creates a Stage that recalls context from memory.

        This Stage retrieves context from the provided memory and either replaces
        the current context or extends it, depending on the specified mode.

        Parameters
        ----------
        memory : Memory[LMMContext, Any]
            The memory instance from which to recall the LMM context.
        handling : Literal["replace", "merge"]
            Determines how the recalled context is used:
            - "replace": Completely replaces the current context with the recalled one.
            - "merge": Merges the recalled context with the current one.
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
                    with ctx.scope("stage.memory_recall"):
                        return await memory.recall()

            case "merge":

                async def stage(
                    *,
                    state: StageState,
                ) -> StageState:
                    with ctx.scope("stage.memory_recall"):
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
        Creates a Stage that stores the current context in memory.

        This Stage saves the current LMM context to the provided memory instance
        without modifying the context or result.

        Parameters
        ----------
        memory : Memory[Any, LMMContext]
            The memory instance in which to store the current LMM context.
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
            with ctx.scope("stage.memory_remember"):
                await memory.remember(state)
                return state

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def completion(
        cls,
        input: Prompt | Multimodal,  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        output: LMMOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Creates a Stage that generates a completion using an LMM.

        This Stage uses the current LMM context along with the provided input to
        generate a new completion. Both the input and the generated completion are
        added to the context, and the completion becomes the new result value.

        The Stage can utilize tools when specified, which may be invoked by the LMM
        during completion generation. Tool calls and their results are automatically
        managed and included in the context.

        Parameters
        ----------
        input : Prompt | Multimodal
            The input content to provide to the LMM, either as a Prompt or Multimodal.
        instruction : Instruction | str | None
            Optional instruction or guidance for the LMM.
        tools : Toolbox | Iterable[Tool] | None
            Optional tools that the LMM can use during completion generation.
        output : LMMOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the LMM invocation.

        Returns
        -------
        Self
            A new Stage instance that generates a completion using the LMM.

        Raises
        ------
        RuntimeError
            If the LMM exceeds the limit of recursive tool calls.

        Examples
        --------
        >>> stage = Stage.completion(
        ...     "Calculate 25% of 80",
        ...     tools=[calculator]
        ... )
        """
        context_extension: LMMContext
        match input:
            case Prompt() as prompt:
                context_extension = prompt.content

            case input_content:
                context_extension = (LMMInput.of(MultimodalContent.of(input_content)),)

        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            with ctx.scope("stage.completion"):
                context, result = await _lmm_completion(
                    instruction=instruction,
                    context=(*state.context, *context_extension),
                    toolbox=toolbox,
                    output=output,
                    **extra,
                )
                return state.updated(
                    context=context,
                    result=result,
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
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        output: LMMOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Creates a Stage that generates a completion using an LMM after prompting for input.

        This Stage awaits the result of an asynchronous function to obtain input content,
        then uses this input along with the current LMM context to generate a completion.
        The obtained input and the generated completion are added to the context, and
        the completion becomes the new result value.

        Similar to the `completion` method, this Stage can utilize tools that may be
        invoked by the LMM during completion generation, with tool calls and their
        results managed automatically.

        Parameters
        ----------
        input : Callable[[], Coroutine[None, None, Multimodal]]
            An async function that returns the input content to provide to the LMM.
            This function will be awaited during stage execution.
        instruction : Instruction | str | None
            Optional instruction or guidance for the LMM.
        tools : Toolbox | Iterable[Tool] | None
            Optional tools that the LMM can use during completion generation.
        output : LMMOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the LMM invocation.

        Returns
        -------
        Self
            A new Stage instance that generates a completion after prompting for input.

        Raises
        ------
        RuntimeError
            If the LMM exceeds the limit of recursive tool calls.

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
            with ctx.scope("stage.completion.prompting"):
                context, result = await _lmm_completion(
                    instruction=instruction,
                    context=(
                        *state.context,
                        LMMInput.of(MultimodalContent.of(await input())),
                    ),
                    toolbox=toolbox,
                    output=output,
                    **extra,
                )
                return state.updated(
                    context=context,
                    result=result,
                )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def loopback_completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        output: LMMOutputSelection = "auto",
        meta: Meta | MetaValues | None = None,
        **extra: Any,
    ) -> Self:
        """
        Creates a Stage that takes the last completion and uses it as the new input.

        This Stage takes the last LMMContext completion, replaces the last input with it,
        and executes a new completion using that input. It's useful for iterative refinement
        of outputs where each completion builds upon the previous one.

        Parameters
        ----------
        instruction : Instruction | str | None
            Optional instruction or guidance for the LMM.
        tools : Toolbox | Iterable[Tool] | None
            Optional tools that the LMM can use during completion generation.
        output : LMMOutputSelection
            Controls the modality/type of the generated completion, default is "auto".
        meta: Meta | MetaValues | None = None
            Additional stage metadata including tags, description etc.
        **extra : Any
            Additional parameters to pass to the LMM invocation.

        Returns
        -------
        Self
            A new Stage instance that performs the loopback completion.

        Raises
        ------
        RuntimeError
            If the LMM exceeds the limit of recursive tool calls.
        """
        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            with ctx.scope("stage.completion.loopback"):
                if not state.context or not isinstance(state.context[-1], LMMCompletion):
                    ctx.log_warning("loopback_completion has been skipped due to invalid context")
                    return state

                # Find the index of the last LMMInput in the context
                last_input_idx: int = len(state.context)
                for idx, element in enumerate(reversed(state.context)):
                    if isinstance(element, LMMInput):
                        last_input_idx = len(state.context) - idx - 1
                        break

                else:
                    # using whole context as is
                    ctx.log_warning("loopback_completion could not find an LMMInput in the context")

                context, result = await _lmm_completion(
                    instruction=instruction,
                    context=(
                        *state.context[:last_input_idx],
                        LMMInput.of(state.context[-1].content),
                    ),
                    toolbox=toolbox,
                    output=output,
                    **extra,
                )
                return state.updated(
                    context=context,
                    result=result,
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
        Creates a Stage that transforms the current result without altering the context.

        This Stage applies a transformation function to the current result to produce a new
        result value, while keeping the LMMContext unchanged.

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
            with ctx.scope("stage.transform.result"):
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
        Creates a Stage that transforms the current context without altering the result.

        This Stage applies a transformation function to the current LMMContext to produce a new
        context, while keeping the result unchanged.

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
            with ctx.scope("stage.transform.context"):
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
        Creates a Stage that trims the current context to a specified limit.

        This Stage reduces the size of the LMMContext by applying a slice or index limit,
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
        Creates a Stage that trims the current context by removing tool calls.

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
                context=tuple(
                    element
                    for element in state.context
                    if not isinstance(element, LMMToolRequests | LMMToolResponses)
                ),
            )

        return cls(
            stage,
            meta=Meta.of(meta),
        )

    @classmethod
    def tool_call[**Args, Result](
        cls,
        tool: FunctionTool[Args, Result],
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Self:
        """
        Creates a Stage that executes a tool and adds its result to the context.

        This Stage calls the provided tool with the given arguments and adds
        the tool call to the LMM context as proper tool request/response pairs.
        This makes the tool interaction visible in the context for subsequent stages.

        Warning:
        --------
        Models usually require tool calls to be between regular input/completion messages.
        You may need to manually adjust context afterwards to ensure proper contents.

        Parameters
        ----------
        tool : FunctionTool[Args, Result]
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
        direct_result: bool
        match tool.handling:
            case "auto":
                direct_result = False

            case "direct":
                direct_result = True

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            with ctx.scope("stage.tool_call"):
                # Create tool request representing the call
                tool_request: LMMToolRequest = LMMToolRequest.of(
                    uuid4().hex,
                    tool=tool.name,
                    arguments=kwargs,
                )

                tool_response: LMMToolResponse
                try:
                    # Execute the tool directly
                    result = await tool(*args, **kwargs)

                    # Create tool response with the result
                    tool_response = LMMToolResponse.of(
                        tool_request.identifier,
                        tool=tool.name,
                        content=MultimodalContent.of(tool.format_result(result)),
                        handling="direct_result" if direct_result else "result",
                    )

                except Exception as exc:
                    tool_response = LMMToolResponse.of(
                        tool_request.identifier,
                        tool=tool.name,
                        content=MultimodalContent.of(tool.format_failure(exc)),
                        handling="direct_result" if direct_result else "error",
                    )

                return state.updated(
                    context=(
                        *state.context,
                        LMMToolRequests.of((tool_request,)),
                        LMMToolResponses.of((tool_response,)),
                    ),
                    result=tool_response.content if direct_result else state.result,
                )

        return cls(
            stage,
            meta=Meta.of({"tool_call": tool.name}),
        )

    @classmethod
    def result_evaluation(
        cls,
        evaluator: PreparedScenarioEvaluator[MultimodalContent]
        | PreparedEvaluator[MultimodalContent],
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Creates a Stage that evaluates the current result using an evaluator.

        This Stage takes the current stage result and runs it through the provided
        evaluator or scenario evaluator. The stage raises StageException when evaluation fails.

        Parameters
        ----------
        evaluator : PreparedScenarioEvaluator[MultimodalContent]
        | PreparedEvaluator[MultimodalContent]
            The evaluator or scenario evaluator to use for evaluation.
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
            with ctx.scope("stage.result_evaluation"):
                evaluation_result: ScenarioEvaluatorResult | EvaluatorResult = await evaluator(
                    state.result
                )

                if evaluation_result.passed:
                    return state  # evaluation passed, keep going

                score: float = evaluation_result.relative_score
                report: str = evaluation_result.report(include_details=__debug__)
                raise StageException(
                    f"Result evaluation failed with relative score: {score * 100:.2f}%",
                    state=state,
                    meta={
                        "evaluation_score": score,
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
        evaluator: PreparedScenarioEvaluator[LMMContext] | PreparedEvaluator[LMMContext],
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Creates a Stage that evaluates the current context using an evaluator.

        This Stage takes the current LMM context and runs it through the provided
        evaluator or scenario evaluator. The stage raises StageException when evaluation fails.

        Parameters
        ----------
        evaluator : PreparedScenarioEvaluator[Value] | PreparedEvaluator[Value]
            The evaluator or scenario evaluator to use for evaluation.
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
            with ctx.scope("stage.context_evaluation"):
                evaluation_result: ScenarioEvaluatorResult | EvaluatorResult = await evaluator(
                    state.context
                )

                if evaluation_result.passed:
                    return state  # evaluation passed, keep going

                score: float = evaluation_result.relative_score
                report: str = evaluation_result.report(include_details=__debug__)
                raise StageException(
                    f"Context evaluation failed with relative score: {score * 100:.2f}%",
                    state=state,
                    meta={
                        "evaluation_score": score,
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
        Creates a Stage that executes another Stage in a loop while a condition is met.

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
                    with ctx.scope("stage.loop"):
                        current_state: StageState = state
                        iteration: int = 0

                        while await condition(state=current_state, iteration=iteration):
                            with ctx.scope("stage.loop.iteration"):
                                for execution in stage_executions:
                                    current_state = await execution(state=current_state)

                                iteration += 1

                        return current_state

            case "after_execution":

                async def stage_loop(
                    *,
                    state: StageState,
                ) -> StageState:
                    with ctx.scope("stage.loop"):
                        current_state: StageState = state
                        iteration: int = 0

                        while True:
                            with ctx.scope("stage.loop.iteration"):
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
        Creates a Stage that executes multiple Stages in sequence.

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
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Creates a Stage that executes multiple Stages concurrently.

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
            with ctx.scope("stage.concurrent"):
                return await merge(
                    branches=cast(  # we are converting errors witin __call__
                        Sequence[StageState | StageException],
                        await gather(
                            *[execution(state=state) for execution in stage_executions],
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
        Creates a Stage that routes execution to one of several stages based on a routing function.

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

        routing = routing if routing is not None else _lmm_routing

        async def router_stage(
            *,
            state: StageState,
        ) -> StageState:
            with ctx.scope("stage.router"):
                selection: str = await routing(
                    state=state,
                    options=options,
                )

            # out of scope - executing selected
            if selected := routes.get(selection):
                return await selected(state=state)

            else:
                raise RuntimeError(
                    "Stage.router selection invalid"
                    f" - missing selection ({selection}) in available options"
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
        **values: MetaValue,
    ) -> Self:
        """
        Creates a copy of this Stage with updated metadata.

        This method allows you to add or update metadata fields for the Stage,
        such as name, description, tags, and other custom metadata values.
        The metadata is particularly important for routing stages where name
        and description are required.

        Parameters
        ----------
        **values : MetaValue
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
        disposables: Disposables | Iterable[Disposable],
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
        disposables: Disposables | Iterable[Disposable] | None = None,
    ) -> Self:
        """
        Creates a copy of this Stage with the specified state context applied.

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

        disposables: Disposables | Iterable[Disposable] | None
            Optional Disposables which will be used for execution of this stage.
            State produced by disposables will be used within the context state.

        Returns
        -------
        Stage
            A new Stage instance that executes within the specified context.
        """
        execution: StageExecution = self._execution

        resolved_disposables: Disposables | None
        match disposables:
            case None:
                resolved_disposables = None

            case Disposables() as disposables:
                resolved_disposables = disposables

            case iterable:
                resolved_disposables = Disposables(*iterable)

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

        return self.__class__(stage, meta=self.meta)

    @overload
    def cached[Key: Hashable](
        self,
        *,
        limit: int | None = None,
        expiration: float | None = None,
        make_key: MakeCacheKey[Key] | None = None,
    ) -> Self: ...

    @overload
    def cached[Key](
        self,
        *,
        make_key: MakeCacheKey[Key],
        read: ReadCache[Key],
        write: WriteCache[Key],
    ) -> Self: ...

    def cached[Key](
        self,
        *,
        limit: int | None = None,
        expiration: float | None = None,
        make_key: MakeCacheKey[Key] | None = None,
        read: ReadCache[Key] | None = None,
        write: WriteCache[Key] | None = None,
    ) -> Self:
        """
        Creates a copy of this Stage with caching enabled.

        The returned Stage will provide the same context and result as the last execution
        if cache keys match.
        Note that the result will be the exact same context and result each time regarless
        of cached stage actual execution assuming no side effects or processing state changes
        are produced within cached stage.

        Parameters
        ----------
        limit : int | None
            Maximum number of entries to store in the cache. If None, cache size is
            one. Ignored when using custom read and write.
        expiration : float | None
            Time in seconds after which cache entries expire. If None, entries never
            expire. Ignored when using custom read and write.
        make_key : MakeCacheKey[Key] | None
            Custom function to generate cache keys from execution inputs. The function
            should accept context and result parameters and return a hashable value
            uniquely identifying the inputs.
        read : ReadCache[Key] | None
            Custom function to retrieve cached values. Must be provided together with
            `write` and `make_key`.
        write : WriteCache[Key] | None
            Custom function to store values in cache. Must be provided together with
            `read` and `make_key`.

        Returns
        -------
        Self
            A new Stage instance with caching enabled.
        """

        if read is not None and write is not None and make_key is not None:
            return self.__class__(
                cast(
                    StageExecution,
                    cache(
                        make_key=make_key,
                        read=read,
                        write=write,
                    )(self._execution),
                ),
                meta=self.meta,
            )

        else:
            assert read is None and write is None and make_key is None  # nosec: B101
            return self.__class__(
                cache(
                    limit=limit,
                    expiration=expiration,
                    make_key=make_key,
                )(self._execution),
                meta=self.meta,
            )

    def with_retry(
        self,
        *,
        limit: int = 1,
        delay: Callable[[int, Exception], float] | float | None = None,
        catching: set[type[Exception]] | tuple[type[Exception], ...] | type[Exception] = Exception,
    ) -> Self:
        """
        Creates a copy of this Stage with retry behavior added.

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
        catching : set[type[Exception]] | tuple[type[Exception], ...] | type[Exception]
            Exception types to catch and retry on. Defaults to catching all Exceptions.

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
        Creates a copy of this Stage with fallback behavior added.

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
                with ctx.scope("stage.fallback"):
                    if exc in exceptions:
                        return await fallback_execution(state=state)

                    else:
                        raise exc

        return self.__class__(stage, meta=self.meta)

    def with_volatile_context(self) -> Self:
        """
        Creates a copy of this Stage that discards context changes.

        The returned Stage will execute normally but will revert any changes
        made to the LMM context when completed, keeping only the result changes.

        Returns
        -------
        Stage
            A new Stage instance that discards context changes.
        """
        execution: StageExecution = self._execution

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            processed_state: StageState = await execution(state=state)
            return processed_state.updated(context=state.context)

        return self.__class__(stage, meta=self.meta)

    def with_volatile_tools_context(self) -> Self:
        """
        Creates a copy of this Stage that discards tool calls added by this stage.

        The returned Stage will execute normally, but upon completion, it will
        remove any LMMToolRequests and LMMToolResponses elements that were added
        to the context during the execution of this specific stage. When produced
        and a previous contexts share a common prefix with tools usage it won't be
        removed from the result context.

        Returns
        -------
        Stage
            A new Stage instance that makes tool-related context elements volatile.
        """
        execution: StageExecution = self._execution

        async def stage(
            *,
            state: StageState,
        ) -> StageState:
            processed_state: StageState = await execution(state=state)

            common_prefix: LMMContext = tuple(
                current
                for previous, current in zip(
                    state.context,
                    processed_state.context,
                    strict=False,
                )
                if current == previous
            )
            striped_suffix: LMMContext = tuple(
                element
                for element in processed_state.context[len(common_prefix) :]
                if not isinstance(element, LMMToolRequests | LMMToolResponses)
            )

            return state.updated(
                context=(*common_prefix, *striped_suffix),
                result=processed_state.result,
            )

        return self.__class__(stage, meta=self.meta)

    def when(
        self,
        condition: StageConditioning | bool,
        /,
        *,
        alternative: Self | None = None,
    ) -> Self:
        """
        Creates a Stage that conditionally executes based on a predicate.

        This method returns a new Stage that will only execute the current Stage
        if the specified condition is met. If the condition is not met and an
        alternative Stage is provided, that alternative will be executed instead.

        Parameters
        ----------
        condition : StageCondition | bool
            A boolean value or function that determines whether this Stage executes.
            If a function is provided, it will be called with the current context and result.
        alternative : Stage | None
            Optional Stage to execute when the condition is not met.

        Returns
        -------
        Stage
            A new Stage instance that applies conditional execution logic.

        Examples
        --------
        >>> # Execute stage only when result contains specific text
        >>> async def has_error(context: LMMContext, result: MultimodalContent):
        ...     return "error" in result.as_string()
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

        return self.__class__(stage, meta=self.meta)

    def ignore_result(self) -> Self:
        """
        Creates a copy of this Stage that ignores its produced result.

        The returned Stage will execute normally but will retain the original
        result value rather than replacing it with the result of execution.

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

        return self.__class__(stage, meta=self.meta)

    def extend_result(self) -> Self:
        """
        Creates a copy of this Stage that extends the current result.

        The returned Stage will execute normally but will append its result
        to the current result rather than replacing it.

        Returns
        -------
        Stage
            A new Stage instance that extends rather than replaces the result.
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

        return self.__class__(stage, meta=self.meta)

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

        except BaseException as exc:
            raise StageException(
                f"Stage execution interrupted: {exc}",
                state=state,
            ) from exc

    async def execute(
        self,
        *state: DataModel | State,
        context: LMMContext | None = None,
        result: MultimodalContent | None = None,
    ) -> MultimodalContent:
        """
        Executes this Stage with the provided initial values.

        This method runs the Stage's processing function with the specified
        initial context, result, and state, and returns the produced result.

        Parameters
        ----------
        *state : DataModel | State
            The initial state models to use.
        context : LMMContext | None
            The initial LMM context to use, or None to use an empty context.
        result : MultimodalContent | None
            The initial result value to use, or None to use an empty result.

        Returns
        -------
        MultimodalContent
            The result produced by executing the Stage.
        """
        with ctx.scope("stage.execution"):
            initial_context: LMMContext
            match context:
                case None:
                    initial_context = ()

                case context:
                    assert not context or isinstance(context[-1], LMMCompletion)  # nosec: B101
                    initial_context = context

            result_state: StageState = await self(
                state=StageState.of(
                    *state,
                    context=initial_context,
                    result=result if result is not None else MultimodalContent.empty,
                )
            )

            return result_state.result


async def _lmm_completion(
    *,
    instruction: Instruction | str | None,
    context: LMMContext,
    toolbox: Toolbox,
    output: LMMOutputSelection,
    **extra: Any,
) -> tuple[LMMContext, MultimodalContent]:
    current_context: LMMContext = context
    formatted_instruction: LMMInstruction | None = Instruction.formatted(instruction)
    repetition_level: int = 0
    while True:
        match await LMM.completion(
            instruction=formatted_instruction,
            context=current_context,
            output=output,
            tools=toolbox.available_tools(repetition_level=repetition_level),
            **extra,
        ):
            case LMMCompletion() as completion:
                return (
                    (*current_context, completion),
                    completion.content,
                )

            case LMMToolRequests() as tool_requests:
                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if direct_results := [
                    response.content
                    for response in tool_responses.responses
                    if response.handling == "direct_result"
                ]:
                    direct_content: MultimodalContent = MultimodalContent.of(*direct_results)

                    return (
                        (
                            *current_context,
                            LMMCompletion.of(direct_content),
                        ),
                        direct_content,
                    )

                else:
                    current_context = (
                        *current_context,
                        tool_requests,
                        tool_responses,
                    )

        repetition_level += 1  # continue tracking repetetive tool calls


async def _lmm_routing(
    *,
    state: StageState,
    options: Mapping[str, Meta],
) -> str:
    options_text = "\n".join(
        [f"- {key}: {meta.description or 'No description'}" for key, meta in options.items()]
    )

    instruction: str = (
        "Based on the conversation context and current result,"  # nosec: B608 - false positive
        " select the most appropriate option from the following:"
        f"\n\n{options_text}"
        "\n\nRespond with with the exact option name within SELECTION xml tag"
        f" like (e.g., '<SELECTION>{next(iter(options.keys()))}</SELECTION>'"
    )

    # Create routing context with the current result as input
    routing_context: LMMContext = (
        *state.context,
        LMMInput.of(
            MultimodalContent.of(
                "<RESULT>",
                state.result,
                "</RESULT>",
            )
        ),
    )

    selection: str
    match await LMM.completion(
        instruction=instruction,
        context=routing_context,
        output="text",
    ):
        case LMMCompletion() as completion:
            if selection_tag := MultimodalTagElement.parse_first(
                "SELECTION",
                content=completion.content,
            ):
                selection = selection_tag.content.to_str().strip().lower()

            else:
                selection = "__INVALID_SELECTION__"

        case other:
            raise RuntimeError(f"LMM routing failed: unexpected result {type(other)}")

    if selection not in options:
        raise ValueError(
            f"LMM routing failed: invalid selection '{selection}'. "
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
