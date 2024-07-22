from collections.abc import Sequence
from typing import Any, Literal, overload

from draive.instructions import Instruction
from draive.lmm.invocation import LMMOutputStream, LMMToolSelection
from draive.lmm.state import LMM
from draive.lmm.tools import ToolSpecification
from draive.scope import ctx
from draive.types import LMMContextElement, LMMOutput

__all__ = [
    "lmm_invocation",
]


@overload
async def lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: Literal[True],
    **extra: Any,
) -> LMMOutputStream: ...


@overload
async def lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMOutput: ...


@overload
async def lmm_invocation(
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: bool,
    **extra: Any,
) -> LMMOutputStream | LMMOutput: ...


async def lmm_invocation(  # noqa: PLR0913
    *,
    instruction: Instruction | str,
    context: Sequence[LMMContextElement],
    tools: Sequence[ToolSpecification] | None = None,
    tool_selection: LMMToolSelection = "auto",
    output: Literal["text", "json"] = "text",
    stream: bool = False,
    **extra: Any,
) -> LMMOutputStream | LMMOutput:
    return await ctx.state(LMM).invocation(
        instruction=instruction,
        context=context,
        tool_selection=tool_selection,
        tools=tools,
        output=output,
        stream=stream,
        **extra,
    )
