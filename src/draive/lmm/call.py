from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm.state import LMMInvocation, LMMStream
from draive.lmm.types import (
    LMMContextElement,
    LMMOutput,
    LMMOutputSelection,
    LMMStreamInput,
    LMMStreamOutput,
    LMMStreamProperties,
    LMMToolSelection,
    LMMToolSpecification,
)
from draive.utils import ConstantStream

__all__ = [
    "lmm_invoke",
    "lmm_stream",
]


async def lmm_invoke(
    *,
    instruction: Instruction | str | None = None,
    context: Sequence[LMMContextElement],
    tool_selection: LMMToolSelection = "auto",
    tools: Iterable[LMMToolSpecification] | None = None,
    output: LMMOutputSelection = "auto",
    **extra: Any,
) -> LMMOutput:
    return await ctx.state(LMMInvocation).invoke(
        instruction=instruction,
        context=context,
        tool_selection=tool_selection,
        tools=tools,
        output=output,
        **extra,
    )


async def lmm_stream(
    *,
    properties: AsyncIterator[LMMStreamProperties] | LMMStreamProperties,
    input: AsyncIterator[LMMStreamInput],  # noqa: A002
    context: Sequence[LMMContextElement] | None = None,
    **extra: Any,
) -> AsyncIterator[LMMStreamOutput]:
    properties_stream: AsyncIterator[LMMStreamProperties]
    match properties:
        case LMMStreamProperties() as constant:
            properties_stream = ConstantStream(constant)

        case variable:
            properties_stream = variable

    return await ctx.state(LMMStream).prepare(
        properties=properties_stream,
        input=input,
        context=context,
        **extra,
    )
