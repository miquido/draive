from asyncio import gather
from collections.abc import Iterable, Sequence
from typing import Any, Self, final, overload

from haiway.context.access import ScopeContext

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMInput,
    LMMOutputSelection,
    LMMToolRequests,
    LMMToolResponses,
)
from draive.lmm.call import lmm_invoke
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import AnyTool, Toolbox
from draive.workflow.types import (
    StageCondition,
    StageContextProcessing,
    StageMerging,
    StageProcessing,
    StageResultProcessing,
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
            context_end_index: int = len(context)
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
                        if toolbox.hide_calls:
                            current_context = (*context[:context_end_index], completion)

                        else:
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
                            if toolbox.hide_calls:
                                current_context = (
                                    *context[:context_end_index],
                                    LMMCompletion.of(direct_content),
                                )

                            else:
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

    @overload
    @classmethod
    def context_trimming(
        cls,
        *,
        limit: int,
    ) -> Self: ...

    @overload
    @classmethod
    def context_trimming(
        cls,
        *,
        slice_limit: slice,
    ) -> Self: ...

    @classmethod
    def context_trimming(
        cls,
        *,
        limit: int | None = None,
        slice_limit: slice | None = None,
    ) -> Self:
        assert limit or slice_limit  # nosec: B101

        if limit:

            async def stage(
                *,
                context: LMMContext,
                result: MultimodalContent,
            ) -> tuple[LMMContext, MultimodalContent]:
                return (context[:limit], result)

        elif slice_limit:

            async def stage(
                *,
                context: LMMContext,
                result: MultimodalContent,
            ) -> tuple[LMMContext, MultimodalContent]:
                return (context[slice_limit], result)

        else:

            async def stage(
                *,
                context: LMMContext,
                result: MultimodalContent,
            ) -> tuple[LMMContext, MultimodalContent]:
                return (context, result)

        return cls(stage)

    @classmethod
    def loop(
        cls,
        stage: Self,
        /,
        *stages: Self,
        condition: StageCondition,
    ) -> Self:
        stage_processings: Sequence[StageProcessing] = tuple(
            stage._processing for stage in (stage, *stages)
        )

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
                for processings in stage_processings:
                    current_context, current_result = await processings(
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

    def conditional(
        self,
        condition: StageCondition,
        /,
        *,
        alternative: Self | None = None,
    ) -> Self:
        processing: StageProcessing = self._processing
        alternative_processing: StageProcessing | None = (
            alternative._processing if alternative else None
        )

        async def stage(
            *,
            context: LMMContext,
            result: MultimodalContent,
        ) -> tuple[LMMContext, MultimodalContent]:
            if await condition(context=context, result=result):
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

    async def __call__(
        self,
        *,
        context: Prompt | LMMContext | None,
    ) -> MultimodalContent:
        current_context: LMMContext
        match context:
            case None:
                current_context = ()

            case Prompt() as prompt:
                current_context = prompt.content

            case context:
                current_context = context

        _, result = await self._processing(
            context=current_context,
            result=MultimodalContent.of(),
        )
        return result
