from collections.abc import AsyncIterator, Iterable
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from haiway import State

from draive.instructions import Instruction
from draive.lmm.tools import ToolSpecification
from draive.parameters import ParametersSpecification
from draive.types import (
    LMMContextElement,
    LMMOutput,
    LMMStreamInput,
    LMMStreamOutput,
)

__all__ = [
    "LMMInvocating",
    "LMMToolSelection",
    "LMMStreamProperties",
]

LMMToolSelection = ToolSpecification | Literal["auto", "required", "none"]


@runtime_checkable
class LMMInvocating(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str | None,
        context: Iterable[LMMContextElement],
        tool_selection: LMMToolSelection,
        tools: Iterable[ToolSpecification] | None,
        output: Literal["auto", "text"] | ParametersSpecification,
        **extra: Any,
    ) -> LMMOutput: ...


class LMMStreamProperties(State):
    instruction: Instruction | str | None = None
    tools: Iterable[ToolSpecification] | None


@runtime_checkable
class LMMStreaming(Protocol):
    async def __call__(
        self,
        *,
        properties: AsyncIterator[LMMStreamProperties],
        input: AsyncIterator[LMMStreamInput],  # noqa: A002
        context: Iterable[LMMContextElement] | None,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...
