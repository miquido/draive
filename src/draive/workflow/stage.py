from asyncio import gather
from collections.abc import Callable, Coroutine, Iterable, Sequence
from typing import Any, Self, final

from haiway import ScopeContext, ctx, traced

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
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import AnyTool, Toolbox
from draive.workflow.types import (
    StageCondition,
    StageContextProcessing,
    StageMerging,
    StageProcessing,
    StageResultProcessing,
    StageStateProcessing,
)

__all__ = [
    "Stage",
]


@final
class Stage:
    @classmethod
    def static(
        cls,
        input: Prompt | Multimodal,  # noqa: A002
        /,
        *,
        completion: Multimodal,
    ) -> Self:
        completion_result = MultimodalContent.of(completion)
        context_extension: LMMContext
        match input:
            case Prompt() as prompt:
                context_extension = (
                    *prompt.content,
                    LMMCompletion.of(completion_result),
                )

            case content:
                context_extension = (
                    LMMInput.of(MultimodalContent.of(content)),
                    LMMCompletion.of(completion_result),
                )

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return ((*context, *context_extension), completion_result)

        return cls(stage)

    @classmethod
    def dynamic(
        cls,
        input: Callable[[], Coroutine[None, None, Prompt | Multimodal]],  # noqa: A002
        /,
        *,
        completion: Callable[[], Coroutine[None, None, Multimodal]],
    ) -> Self:
        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            completion_result = MultimodalContent.of(await completion())
            context_extension: LMMContext
            match await input():
                case Prompt() as prompt:
                    context_extension = (
                        *prompt.content,
                        LMMCompletion.of(completion_result),
                    )

                case content:
                    context_extension = (
                        LMMInput.of(MultimodalContent.of(content)),
                        LMMCompletion.of(completion_result),
                    )

            return ((*context, *context_extension), completion_result)

        return cls(stage)

    @classmethod
    def completion(
        cls,
        input: Prompt | Multimodal,  # noqa: A002
        /,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
        **extra: Any,
    ) -> Self:
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
            current_context: LMMContext = (*context, *context_extension)
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
                            response.content
                            for response in tool_responses.responses
                            if response.direct
                        ]:
                            direct_content: MultimodalContent = MultimodalContent.of(
                                *direct_results
                            )

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

        return cls(stage)

    @classmethod
    def loopback_completion(
        cls,
        instruction: Instruction | str | None = None,
        tools: Toolbox | Iterable[AnyTool] | None = None,
        output: LMMOutputSelection = "auto",
        **extra: Any,
    ) -> Self:
        toolbox: Toolbox = Toolbox.of(tools)

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            if not context or not isinstance(context[-1], LMMCompletion):
                ctx.log_warning("loopback_completion has been skipped due to invalid context")
                return (context, result)

            # skipping meta as it is no longer applicable to input converted from output
            current_context: LMMContext = (*context[:-2], LMMInput.of(context[-1].content))
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
                            response.content
                            for response in tool_responses.responses
                            if response.direct
                        ]:
                            direct_content: MultimodalContent = MultimodalContent.of(
                                *direct_results
                            )

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

        return cls(stage)

    @classmethod
    def result_processing(
        cls,
        processing: StageResultProcessing,
        /,
    ) -> Self:
        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return (context, await processing(result))

        return cls(stage)

    @classmethod
    def context_processing(
        cls,
        processing: StageContextProcessing,
        /,
    ) -> Self:
        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return (await processing(context), result)

        return cls(stage)

    @classmethod
    def context_trimming(
        cls,
        *,
        limit: slice | int | None = None,
    ) -> Self:
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
                    return (context[:index], result)

            case slice() as index_slice:

                async def stage(
                    *,
                    context: LMMContext,
                    result: MultimodalContent,
                ) -> tuple[LMMContext, MultimodalContent]:
                    return (context[index_slice], result)

        return cls(stage)

    @classmethod
    def state_processing(
        cls,
        processing: StageStateProcessing,
        /,
    ) -> Self:
        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            await processing(context, result)
            return (context, result)

        return cls(stage)

    @classmethod
    def loop(
        cls,
        stage: Self,
        /,
        condition: StageCondition,
    ) -> Self:
        stage_processing: StageProcessing = stage._processing

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

            return (current_context, current_result)

        return cls(stage_loop)

    @classmethod
    def sequence(
        cls,
        stage: Self,
        /,
        *stages: Self,
    ) -> Self:
        stage_processings: Sequence[StageProcessing] = tuple(
            stage._processing for stage in (stage, *stages)
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
        stage_processings: Sequence[StageProcessing] = tuple(
            stage._processing for stage in (stage, *stages)
        )

        async def concurrent_stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            return await merge(
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

        return cls(concurrent_stage)

    def __init__(
        self,
        processing: StageProcessing,
        /,
    ) -> None:
        assert not isinstance(processing, Stage)  # nosec: B101
        assert isinstance(processing, StageProcessing)  # nosec: B101
        self._processing: StageProcessing
        object.__setattr__(
            self,
            "_processing",
            processing,
        )

    def with_execution_context(
        self,
        execution_context: ScopeContext,
        /,
    ) -> Self:
        processing: StageProcessing = self._processing

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            async with execution_context:
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
        processing: StageProcessing = self._processing

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

    def with_volatile_context(self) -> Self:
        processing: StageProcessing = self._processing

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

    def conditional(
        self,
        condition: StageCondition | bool,
        /,
        *,
        alternative: Self | None = None,
    ) -> Self:
        processing: StageProcessing = self._processing
        alternative_processing: StageProcessing | None = (
            alternative._processing if alternative else None
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

    def skipping_result(self) -> Self:
        processing: StageProcessing = self._processing

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

    def extending_result(self) -> Self:
        processing: StageProcessing = self._processing

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            processed_context, processed_result = await processing(
                context=context,
                result=result,
            )
            return (processed_context, result.appending(processed_result))

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
        context: Prompt | LMMContext | None = None,
    ) -> MultimodalContent:
        match context:
            case None:
                return await self.__call__(context=())

            case Prompt() as prompt:
                return await self.__call__(context=prompt.content)

            case context:
                return await self.__call__(context=context)

    async def __call__(
        self,
        *,
        context: LMMContext,
    ) -> MultimodalContent:
        _, result = await self._processing(
            context=context,
            result=MultimodalContent.of(),
        )
        return result
