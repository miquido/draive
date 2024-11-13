from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm.state import LMMInvocation, LMMStream
from draive.lmm.types import (
    LMMContextElement,
    LMMOutput,
    LMMStreamInput,
    LMMStreamOutput,
    LMMStreamProperties,
    LMMToolSelection,
    ToolSpecification,
)
from draive.parameters import ParametersSpecification
from draive.utils import ConstantStream

__all__ = [
    "lmm_invoke",
    "lmm_stream",
]


async def lmm_invoke(
    *,
    instruction: Instruction | str | None = None,
    context: Iterable[LMMContextElement],
    tool_selection: LMMToolSelection = "auto",
    tools: Iterable[ToolSpecification] | None = None,
    output: Literal["auto", "text"] | ParametersSpecification = "auto",
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
    context: Iterable[LMMContextElement] | None = None,
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
