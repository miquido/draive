from asyncio import gather
from collections.abc import Callable, Coroutine, Iterable, Sequence
from typing import Any, ClassVar, Literal, Self, final, overload

from haiway import ScopeContext, State, StateContext, ctx, retry, traced

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMInput,
    LMMOutputSelection,
    LMMToolRequests,
    LMMToolResponses,
    lmm_invoke,
)
from draive.lmm.types import LMMContextElement
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters.model import DataModel
from draive.prompts import Prompt
from draive.stages.types import (
    StageCondition,
    StageContextTransforming,
    StageException,
    StageMerging,
    StageProcessing,
    StageResultTransforming,
    StageStateAccessing,
)
from draive.tools import AnyTool, Toolbox
from draive.utils import ProcessingState
from draive.utils.memory import Memory
from draive.utils.processing import Processing

__all__ = [
    "Stage",
]


@final
class Stage:
    """
    Encapsulates a unit of work within a LMM context.

    A Stage represents a transformation function that processes an LMM context and
    result values to produce updated versions of both. Stages can be composed
    to create complex processing pipelines while maintaining precise control over
    the LMM context used at each step.

    Each stage accepts an LMM context and a result value as input, and returns
    a tuple containing the updated context and result. The stage's operation may
    modify either or both of these values.

    Stages support various composition patterns including sequencing, conditional
    execution, looping, and concurrent execution with result merging.
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
    ) -> Self: ...

    @overload
    @classmethod
    def predefined(
        cls,
        element: Prompt | Multimodal,
        /,
        *elements: Multimodal,
    ) -> Self: ...

    @classmethod
    def predefined(
        cls,
        element: Prompt | LMMContextElement | Multimodal,
        /,
        *elements: LMMContextElement | Multimodal,
    ) -> Self:
        """
        Creates a Stage that inserts predefined elements into the context.

        This Stage adds the specified elements to the LMM context and
        sets the result to the last provided completion value. It's useful for
        injecting static content into a processing pipeline.
        Note that context must always end with LMMCompletion as the result.

        Parameters
        ----------
        element : Prompt | LMMContextElement | Multimodal
            The first element to add to the context.
        *elements : LMMContextElement | Multimodal
            Additional elements to add to the context.

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
            if idx % 2 == 0:
                if isinstance(value, LMMContextElement):
                    context.append(value)

                else:
                    context.append(LMMInput.of(MultimodalContent.of(value)))

            elif isinstance(value, LMMContextElement):
                context.append(value)

            else:
                context.append(LMMCompletion.of(MultimodalContent.of(value)))

        context_extension: LMMContext = tuple(context)
        assert isinstance(context_extension[-1], LMMCompletion)  # nosec: B101
        completion_result: MultimodalContent = context_extension[-1].content

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return ((*context, *context_extension), completion_result)

        return cls(stage)

    @classmethod
    def memory_recall(
        cls,
        memory: Memory[LMMContext, Any],
        /,
        *,
        handling: Literal["replace", "extend"] = "replace",
    ) -> Self:
        """
        Creates a Stage that recalls context from memory.

        This Stage retrieves context from the provided memory and either replaces
        the current context or extends it, depending on the specified mode.

        Parameters
        ----------
        memory : Memory[LMMContext, Any]
            The memory instance from which to recall the LMM context.
        handling : Literal["replace", "extend"]
            Determines how the recalled context is used:
            - "replace": Completely replaces the current context with the recalled one.
            - "extend": Appends the recalled context to the current one.
            Default is "replace".

        Returns
        -------
        Self
            A new Stage instance that recalls context from memory.
        """

        match handling:
            case "replace":

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    return (await memory.recall(), result)

            case "extend":

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    return ((*context, *await memory.recall()), result)

        return cls(stage)

    @classmethod
    def memory_remember(
        cls,
        memory: Memory[Any, LMMContext],
        /,
    ) -> Self:
        """
        Creates a Stage that stores the current context in memory.

        This Stage saves the current LMM context to the provided memory instance
        without modifying the context or result.

        Parameters
        ----------
        memory : Memory[Any, LMMContext]
            The memory instance in which to store the current LMM context.

        Returns
        -------
        Self
            A new Stage instance that remembers context in memory.
        """

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            await memory.remember(context)
            return (context, result)

        return cls(stage)

    @classmethod
    def completion(
        cls,
        input: Prompt | Multimodal,  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
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
        instruction : Instruction | str | None, optional
            Optional instruction or guidance for the LMM.
        tools : Toolbox | Iterable[AnyTool] | None, optional
            Optional tools that the LMM can use during completion generation.
        output : LMMOutputSelection, optional
            Controls the modality/type of the generated completion, default is "auto".
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
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return await _lmm_completion(
                instruction=instruction,
                context=(*context, *context_extension),
                toolbox=toolbox,
                output=output,
                **extra,
            )

        return cls(stage)

    @classmethod
    def prompting_completion(
        cls,
        input: Callable[[], Coroutine[None, None, Multimodal]],  # noqa: A002
        /,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
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
        instruction : Instruction | str | None, optional
            Optional instruction or guidance for the LMM.
        tools : Toolbox | Iterable[AnyTool] | None, optional
            Optional tools that the LMM can use during completion generation.
        output : LMMOutputSelection, optional
            Controls the modality/type of the generated completion, default is "auto".
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
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return await _lmm_completion(
                instruction=instruction,
                context=(*context, LMMInput.of(MultimodalContent.of(await input()))),
                toolbox=toolbox,
                output=output,
                **extra,
            )

        return cls(stage)

    @classmethod
    def loopback_completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
        **extra: Any,
    ) -> Self:
        """
        Creates a Stage that takes the last completion and uses it as the new input.

        This Stage takes the last LMMContext completion, replaces the last input with it,
        and executes a new completion using that input. It's useful for iterative refinement
        of outputs where each completion builds upon the previous one.

        Parameters
        ----------
        instruction : Instruction | str | None, optional
            Optional instruction or guidance for the LMM.
        tools : Toolbox | Iterable[AnyTool] | None, optional
            Optional tools that the LMM can use during completion generation.
        output : LMMOutputSelection, optional
            Controls the modality/type of the generated completion, default is "auto".
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
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            if not context or not isinstance(context[-1], LMMCompletion):
                ctx.log_warning("loopback_completion has been skipped due to invalid context")
                return (context, result)

            return await _lmm_completion(
                instruction=instruction,
                # skipping meta as it is no longer applicable to input converted from output
                context=(*context[:-2], LMMInput.of(context[-1].content)),
                toolbox=toolbox,
                output=output,
                **extra,
            )

        return cls(stage)

    @classmethod
    def transform_result(
        cls,
        transformation: StageResultTransforming,
        /,
    ) -> Self:
        """
        Creates a Stage that transforms the current result without altering the context.

        This Stage applies a transformation function to the current result to produce a new
        result value, while keeping the LMMContext unchanged.

        Parameters
        ----------
        transformation : StageResultTransforming
            Function that takes the current result and returns a transformed result.

        Returns
        -------
        Self
            A new Stage instance that transforms the result.
        """

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return (context, await transformation(result))

        return cls(stage)

    @classmethod
    def transform_context(
        cls,
        transformation: StageContextTransforming,
        /,
    ) -> Self:
        """
        Creates a Stage that transforms the current context without altering the result.

        This Stage applies a transformation function to the current LMMContext to produce a new
        context, while keeping the result unchanged.

        Parameters
        ----------
        transformation : StageContextTransforming
            Function that takes the current context and returns a transformed context.

        Returns
        -------
        Self
            A new Stage instance that transforms the context.
        """

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            transformed_context: LMMContext = await transformation(context)
            assert not transformed_context or isinstance(transformed_context[-1], LMMCompletion)  # nosec: B101
            return (transformed_context, result)

        return cls(stage)

    @classmethod
    def trim_context(
        cls,
        *,
        limit: slice | int | None = None,
    ) -> Self:
        """
        Creates a Stage that trims the current context to a specified limit.

        This Stage reduces the size of the LMMContext by applying a slice or index limit,
        or by clearing it entirely if no limit is specified.

        Parameters
        ----------
        limit : slice | int | None, optional
            Specifies how to trim the context:
            - None: Clear the context completely
            - int: Keep only the first N elements
            - slice: Apply a slice operation to the context

        Returns
        -------
        Self
            A new Stage instance that trims the context.
        """
        match limit:
            case None:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    return ((), result)

            case int() as index:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    trimmed_context: LMMContext = context[:index]
                    assert not trimmed_context or isinstance(trimmed_context[-1], LMMCompletion)  # nosec: B101
                    return (trimmed_context, result)

            case slice() as index_slice:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    trimmed_context: LMMContext = context[index_slice]
                    assert not trimmed_context or isinstance(trimmed_context[-1], LMMCompletion)  # nosec: B101
                    return (trimmed_context, result)

        return cls(stage)

    @classmethod
    def strip_context_tools(cls) -> Self:
        """
        Creates a Stage that trims the current context by removing tool calls.

        Returns
        -------
        Self
            A new Stage instance that removes tool-related elements from the context.
        """

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return (
                tuple(
                    element
                    for element in context
                    if not isinstance(element, LMMToolRequests | LMMToolResponses)
                ),
                result,
            )

        return cls(stage)

    @classmethod
    def access_state(
        cls,
        access: StageStateAccessing,
        /,
    ) -> Self:
        """
        Creates a Stage that allows access to the state without changing context or result.

        This Stage enables interaction with the contextual state through the provided
        processing function, allowing state to be read or written without modifying
        the LMM context or result.

        Parameters
        ----------
        access : StageStateAccessing
            A function that accesses the state. It should be an async function
            that accepts the current context and result but doesn't need to return anything.

        Returns
        -------
        Stage
            A new Stage instance that provides access to the state.
        """

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            await access(context, result)
            return (context, result)

        return cls(stage)

    @classmethod
    def loop(
        cls,
        stage: Self,
        /,
        *,
        condition: StageCondition,
    ) -> Self:
        """
        Creates a Stage that executes another Stage in a loop while a condition is met.

        This Stage repeatedly executes the provided Stage as long as the condition
        function returns True when applied to the current context and result.

        Parameters
        ----------
        stage : Stage
            The Stage to execute repeatedly.
        condition : StageCondition
            A function that determines whether to continue looping. It should be
            an async function that accepts the current context and result and returns
            a boolean.

        Returns
        -------
        Stage
            A new Stage instance that implements the looping behavior.
        """
        stage_processing: StageProcessing = stage.processing

        async def stage_loop(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            current_context: LMMContext = context
            current_result: MultimodalContent = result
            while await condition(
                context=current_context,
                result=current_result,
            ):
                current_context, current_result = await stage_processing(
                    context=current_context,
                    result=current_result,
                )
                assert not current_context or isinstance(current_context[-1], LMMCompletion)  # nosec: B101

            return (current_context, current_result)

        return cls(stage_loop)

    @classmethod
    def sequence(
        cls,
        stage: Self,
        /,
        *stages: Self,
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

        Returns
        -------
        Stage
            A new Stage instance that executes the provided Stages sequentially.
        """
        stage_processings: Sequence[StageProcessing] = tuple(
            stage.processing for stage in (stage, *stages)
        )

        async def stage_sequence(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            current_context: LMMContext = context
            current_result: MultimodalContent = result
            for processings in stage_processings:
                current_context, current_result = await processings(
                    context=current_context,
                    result=current_result,
                )
                assert not current_context or isinstance(current_context[-1], LMMCompletion)  # nosec: B101

            return (current_context, current_result)

        return cls(stage_sequence)

    @classmethod
    def concurrent(
        cls,
        stage: Self,
        /,
        *stages: Self,
        merge: StageMerging,
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

        Returns
        -------
        Stage
            A new Stage instance that executes Stages concurrently and merges results.
        """
        stage_processings: Sequence[StageProcessing] = tuple(
            stage.processing for stage in (stage, *stages)
        )

        async def concurrent_stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            merged_context: LMMContext
            merged_result: MultimodalContent
            merged_context, merged_result = await merge(
                branches=await gather(
                    *[
                        processing(
                            context=context,
                            result=result,
                        )
                        for processing in stage_processings
                    ],
                    return_exceptions=True,
                )
            )
            assert not merged_context or isinstance(merged_context[-1], LMMCompletion)  # nosec: B101
            return (merged_context, merged_result)

        return cls(concurrent_stage)

    __slots__ = ("processing",)

    def __init__(
        self,
        processing: StageProcessing,
        /,
    ) -> None:
        assert not isinstance(processing, Stage)  # nosec: B101
        assert isinstance(processing, StageProcessing)  # nosec: B101
        self.processing: StageProcessing
        object.__setattr__(
            self,
            "processing",
            processing,
        )

    def with_execution_context(
        self,
        execution_context: ScopeContext | StateContext,
        /,
    ) -> Self:
        """
        Creates a copy of this Stage with the specified execution context applied.

        The returned Stage will execute within the provided execution context,
        which affects how contextual state is accessed and modified.

        Parameters
        ----------
        execution_context : ScopeContext | StateContext
            The execution context to apply to the Stage.

        Returns
        -------
        Stage
            A new Stage instance that executes within the specified context.
        """
        processing: StageProcessing = self.processing

        match execution_context:
            case StateContext() as state_context:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    with state_context:
                        return await processing(
                            context=context,
                            result=result,
                        )

            case ScopeContext() as scope_context:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    async with scope_context:
                        return await processing(
                            context=context,
                            result=result,
                        )

        return self.__class__(stage)

    def traced(
        self,
        *,
        label: str,
    ) -> Self:
        """
        Creates a copy of this Stage with tracing enabled.

        The returned Stage will emit trace information during execution,
        labeled with the provided label.

        Parameters
        ----------
        label : str
            The label to use for identifying this Stage in traces.

        Returns
        -------
        Stage
            A new Stage instance with tracing enabled.
        """
        processing: StageProcessing = self.processing

        @traced(label=label)
        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return await processing(
                context=context,
                result=result,
            )

        return self.__class__(stage)

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
        limit : int, optional
            Maximum number of retry attempts. Defaults to 1.
        delay : Callable[[int, Exception], float] | float | None, optional
            Delay between retries in seconds. Can be a fixed float, a function
            that accepts the retry attempt number (starting from 0) and the raised
            exception and returns the delay, or None for no delay. Defaults to None.
        catching : set[type[Exception]] | tuple[type[Exception], ...] | type[Exception], optional
            Exception types to catch and retry on. Defaults to catching all Exceptions.

        Returns
        -------
        Self
            A new Stage instance with retry behavior.
        """
        processing: StageProcessing = self.processing

        @retry(
            limit=limit,
            delay=delay,
            catching=catching,
        )
        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return await processing(
                context=context,
                result=result,
            )

        return self.__class__(stage)

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
        processing: StageProcessing = self.processing

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            _, processed_result = await processing(
                context=context,
                result=result,
            )
            return (context, processed_result)

        return self.__class__(stage)

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
        processing: StageProcessing = self.processing

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            processed_context, processed_result = await processing(
                context=context,
                result=result,
            )

            common_prefix: LMMContext = tuple(
                current
                for previous, current in zip(
                    context,
                    processed_context,
                    strict=True,
                )
                if current == previous
            )
            striped_suffix: LMMContext = tuple(
                element
                for element in processed_context[len(common_prefix) :]
                if not isinstance(element, LMMToolRequests | LMMToolResponses)
            )

            merged_context: LMMContext = (*common_prefix, *striped_suffix)
            assert not merged_context or isinstance(merged_context[-1], LMMCompletion)  # nosec: B101
            return (merged_context, processed_result)

        return self.__class__(stage)

    def when(
        self,
        condition: StageCondition | bool,
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
        alternative : Stage | None, optional
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
        processing: StageProcessing = self.processing
        alternative_processing: StageProcessing | None = (
            alternative.processing if alternative else None
        )

        match condition:
            case bool() as value:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    if value:
                        return await processing(
                            context=context,
                            result=result,
                        )

                    elif alternative_processing:
                        return await alternative_processing(
                            context=context,
                            result=result,
                        )

                    else:
                        return (context, result)

            case function:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    if await function(context=context, result=result):
                        return await processing(
                            context=context,
                            result=result,
                        )

                    elif alternative_processing:
                        return await alternative_processing(
                            context=context,
                            result=result,
                        )

                    else:
                        return (context, result)

        return self.__class__(stage)

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
        processing: StageProcessing = self.processing

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            processed_context, _ = await processing(
                context=context,
                result=result,
            )

            return (processed_context, result)

        return self.__class__(stage)

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
        processing: StageProcessing = self.processing

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            processed_context, processed_result = await processing(
                context=context,
                result=result,
            )
            return (processed_context, result.extending(processed_result))

        return self.__class__(stage)

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

    async def execute(
        self,
        *,
        context: LMMContext | None = None,
        result: MultimodalContent | None = None,
        state: Iterable[DataModel | State] | None = None,
    ) -> MultimodalContent:
        """
        Executes this Stage with the provided initial values.

        This method runs the Stage's processing function with the specified
        initial context, result, and state, and returns the produced result.

        Parameters
        ----------
        context : Prompt | LMMContext | None, optional
            The initial LMM context to use, or None to use an empty context.
        result : MultimodalContent | None, optional
            The initial result value to use, or None to use an empty result.
        state : Iterable[DataModel | State] | None, optional
            The initial state models to use, or None to use an empty state.

        Returns
        -------
        MultimodalContent
            The result produced by executing the Stage.
        """
        stage_state = ProcessingState(state)
        with ctx.updated(
            ctx.state(Processing).updated(
                # keep current processing unchanged
                # but use local state for execution
                state_reading=stage_state.read,
                state_writing=stage_state.write,
            )
        ):
            initial_context: LMMContext
            match context:
                case None:
                    initial_context = ()

                case context:
                    assert not context or isinstance(context[-1], LMMCompletion)  # nosec: B101
                    initial_context = context

            try:
                _, processed = await self.processing(
                    context=initial_context,
                    result=result if result is not None else MultimodalContent.empty,
                )

            except StageException as exc:
                if exc.execution_result is not None:
                    return exc.execution_result

                else:
                    raise exc

        return processed


async def _lmm_completion(
    *,
    instruction: Instruction | str | None,
    context: LMMContext,
    toolbox: Toolbox,
    output: LMMOutputSelection,
    **extra: Any,
) -> tuple[LMMContext, MultimodalContent]:
    current_context: LMMContext = context
    recursion_level: int = 0
    while recursion_level <= toolbox.repeated_calls_limit:
        match await lmm_invoke(
            instruction=instruction,
            context=current_context,
            output=output,
            tools=toolbox.available_tools(),
            tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
            **extra,
        ):
            case LMMCompletion() as completion:
                current_context = (*current_context, completion)

                return (current_context, completion.content)

            case LMMToolRequests() as tool_requests:
                tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                if direct_results := [
                    response.content for response in tool_responses.responses if response.direct
                ]:
                    direct_content: MultimodalContent = MultimodalContent.of(*direct_results)

                    current_context = (
                        *current_context,
                        LMMCompletion.of(direct_content),
                    )

                    return (current_context, direct_content)

                else:
                    current_context = (
                        *current_context,
                        tool_requests,
                        tool_responses,
                    )

        recursion_level += 1  # continue with next recursion level

    raise RuntimeError("LMM exceeded limit of recursive calls")


@Stage
async def _noop(
    *,
    context: LMMContext,
    result: MultimodalContent,
) -> tuple[LMMContext, MultimodalContent]:
    return (context, result)


Stage.noop = _noop
