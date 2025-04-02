from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal, final, overload

from haiway import State, ctx

from draive.instructions import Instruction
from draive.lmm.types import (
    LMMCompleting,
    LMMContext,
    LMMOutput,
    LMMOutputSelection,
    LMMSessionPreparing,
    LMMSessionProperties,
    LMMStreamInput,
    LMMStreamOutput,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.utils import ConstantStream

__all__ = [
    "LMM",
    "LMMSession",
]


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
        properties: AsyncIterator[LMMSessionProperties] | LMMSessionProperties,
        input: AsyncIterator[LMMStreamInput],  # noqa: A002
        context: LMMContext | None = None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]:
        properties_stream: AsyncIterator[LMMSessionProperties]
        match properties:
            case LMMSessionProperties() as constant:
                properties_stream = ConstantStream(constant)

            case iterable:
                properties_stream = iterable

        return await ctx.state(cls).preparing(
            properties=properties_stream,
            input=input,
            context=context,
            **extra,
        )

    preparing: LMMSessionPreparing
