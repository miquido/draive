from collections.abc import Iterable
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from draive.instructions import Instruction
from draive.lmm.tools import ToolSpecification
from draive.parameters import ParametersSpecification
from draive.types import LMMContextElement, LMMOutput, MultimodalContent

__all__ = [
    "LMMInvocation",
    "LMMToolSelection",
]

LMMToolSelection = ToolSpecification | Literal["auto", "required", "none"]


@runtime_checkable
class LMMInvocation(Protocol):
    async def __call__(  # noqa: PLR0913
        self,
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        prefill: MultimodalContent | None,
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        **extra: Any,
    ) -> LMMOutput: ...
