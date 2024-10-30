from collections.abc import Iterable
from typing import Any, Literal

from haiway import ctx

from draive.instructions import Instruction
from draive.lmm.state import LMM
from draive.lmm.tools import ToolSpecification
from draive.lmm.types import LMMToolSelection
from draive.parameters import ParametersSpecification
from draive.types import LMMContextElement, LMMOutput, MultimodalContent

__all__ = [
    "lmm_invocation",
]


async def lmm_invocation(  # noqa: PLR0913
    *,
    instruction: Instruction | str | None = None,
    context: Iterable[LMMContextElement],
    prefill: MultimodalContent | None = None,
    tool_selection: LMMToolSelection = "auto",
    tools: Iterable[ToolSpecification] | None = None,
    output: Literal["auto", "text"] | ParametersSpecification = "auto",
    **extra: Any,
) -> LMMOutput:
    return await ctx.state(LMM).invocation(
        instruction=instruction,
        context=context,
        prefill=prefill,
        tool_selection=tool_selection,
        tools=tools,
        output=output,
        **extra,
    )
