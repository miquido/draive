from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from typing import Any, Self, final, overload
from uuid import uuid4

from haiway import as_list
from haiway.context.access import ScopeContext

from draive.instructions import Instruction
from draive.lmm import (
    LMMCompletion,
    LMMContext,
    LMMContextElement,
    LMMInput,
    LMMOutputSelection,
    LMMToolRequest,
    LMMToolRequests,
    LMMToolResponses,
)
from draive.lmm.call import lmm_invoke
from draive.multimodal import Multimodal, MultimodalContent
from draive.prompts import Prompt
from draive.tools import AnyTool, Tool, Toolbox
from draive.workflow.types import (
    StageCondition,
    StageContextProcessing,
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
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            context.extend(context_extension)

            return completion_result

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
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            context.extend(context_extension)
            recursion_level: int = 0
            context_end_index: int = len(context)
            while recursion_level <= toolbox.repeated_calls_limit:
                match await lmm_invoke(
                    instruction=instruction,
                    context=context,
                    output=output,
                    tools=toolbox.available_tools(),
                    tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
                    **extra,
                ):
                    case LMMCompletion() as completion:
                        if toolbox.hide_calls:
                            del context[context_end_index:]  # remove tool calls from context

                        context.append(completion)
                        return completion.content

                    case LMMToolRequests() as tool_requests:
                        tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                        if direct_results := [
                            response.content
                            for response in tool_responses.responses
                            if response.direct
                        ]:
                            if toolbox.hide_calls:
                                del context[context_end_index:]  # remove tool calls from context
                            direct_content: MultimodalContent = MultimodalContent.of(
                                *direct_results
                            )
                            context.append(LMMCompletion.of(direct_content))
                            return direct_content

                        else:
                            context.extend(
                                [
                                    tool_requests,
                                    tool_responses,
                                ]
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
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            return await processing(result)

        return cls(stage)

    @classmethod
    def context_processing(
        cls,
        processing: StageContextProcessing,
        /,
    ) -> Self:
        async def stage(
            *,
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            await processing(context)

            return result

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
                context: MutableSequence[LMMContextElement],
                result: MultimodalContent,
            ) -> MultimodalContent:
                context[:limit]

                return result

        elif slice_limit:

            async def stage(
                *,
                context: MutableSequence[LMMContextElement],
                result: MultimodalContent,
            ) -> MultimodalContent:
                context[slice_limit]

                return result

        else:

            async def stage(
                *,
                context: MutableSequence[LMMContextElement],
                result: MultimodalContent,
            ) -> MultimodalContent:
                return result

        return cls(stage)

    @classmethod
    def tool_call[**Args, Result](
        cls,
        tool: Tool[Args, Result],
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Self:
        assert not args  # nosec: B101
        toolbox: Toolbox = Toolbox.of(tool)
        arguments: Mapping[str, Any] = kwargs

        async def stage(
            *,
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            tool_requests = LMMToolRequests(
                requests=[
                    LMMToolRequest(
                        identifier=uuid4().hex,
                        tool=tool.name,
                        arguments=arguments,
                    )
                ],
            )
            tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)
            context.extend(
                (
                    tool_requests,
                    tool_responses,
                )
            )

            if direct_responses := [
                response for response in tool_responses.responses if response.direct
            ]:
                return MultimodalContent.of(*[response.content for response in direct_responses])

            else:
                return result

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
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            current_result: MultimodalContent = result
            while await condition(
                context=context,
                result=result,
            ):
                for processings in stage_processings:
                    current_result = await processings(
                        context=context,
                        result=result,
                    )

            return current_result

        return cls(stage_loop)

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
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
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
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
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
                return result

        return self.__class__(stage)

    def skipping_result(self) -> Self:
        processing: StageProcessing = self._processing

        async def stage(
            *,
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            await processing(
                context=context,
                result=result,
            )

            return result

        return self.__class__(stage)

    def extending_result(self) -> Self:
        processing: StageProcessing = self._processing

        async def stage(
            *,
            context: MutableSequence[LMMContextElement],
            result: MultimodalContent,
        ) -> MultimodalContent:
            return result.appending(
                await processing(
                    context=context,
                    result=result,
                )
            )

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
        result: MultimodalContent,
    ) -> MultimodalContent:
        mutable_context: MutableSequence[LMMContextElement]
        match context:
            case None:
                mutable_context = []

            case Prompt() as prompt:
                mutable_context = as_list(prompt.content)

            case context:
                mutable_context = as_list(context)

        return await self._processing(
            context=mutable_context,
            result=result,
        )
