from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any, Literal, final, overload

from haiway import State, ctx

from draive.instructions import Instruction
from draive.lmm.types import (
    LMMCompleting,
    LMMContext,
    LMMOutput,
    LMMOutputSelection,
    LMMSessionOutput,
    LMMSessionOutputSelection,
    LMMSessionPreparing,
    LMMStreamInput,
    LMMStreamOutput,
    LMMToolSelection,
    LMMToolSpecification,
)

__all__ = (
    "LMM",
    "LMMSession",
)


@final
class LMM(State):
    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        context: LMMContext,
        tool_selection: LMMToolSelection = "auto",
        tools: Iterable[LMMToolSpecification] | None = None,
        output: LMMOutputSelection = "auto",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        context: LMMContext,
        tool_selection: LMMToolSelection = "auto",
        tools: Iterable[LMMToolSpecification] | None = None,
        output: LMMOutputSelection = "auto",
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    @classmethod
    async def completion(
        cls,
        *,
        instruction: Instruction | str | None = None,
        context: LMMContext,
        tool_selection: LMMToolSelection = "auto",
        tools: Iterable[LMMToolSpecification] | None = None,
        output: LMMOutputSelection = "auto",
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            return await ctx.state(cls).completing(
                instruction=Instruction.of(instruction),
                context=context,
                tool_selection=tool_selection,
                tools=tools,
                output=output,
                stream=True,
                **extra,
            )

        else:
            return await ctx.state(cls).completing(
                instruction=Instruction.of(instruction),
                context=context,
                tool_selection=tool_selection,
                tools=tools,
                output=output,
                **extra,
            )

    completing: LMMCompleting


@final
class LMMSession(State):
    @classmethod
    async def prepare(
        cls,
        *,
        instruction: Instruction | None = None,
        initial_context: LMMContext | None = None,
        input_stream: AsyncIterator[LMMStreamInput],
        output: LMMSessionOutputSelection = "auto",
        tools: Sequence[LMMToolSpecification] | None = None,
        tool_selection: LMMToolSelection = "auto",
        **extra: Any,
    ) -> AsyncIterator[LMMSessionOutput]:
        return await ctx.state(cls).preparing(
            instruction=instruction,
            initial_context=initial_context,
            input_stream=input_stream,
            output=output,
            tools=tools if tools is not None else (),
            tool_selection=tool_selection,
            **extra,
        )

    preparing: LMMSessionPreparing
