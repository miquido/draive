from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

from draive.parameters import ToolSpecification
from draive.types import LMMContextElement, LMMOutput, LMMOutputStream

__all__ = [
    "LMMInvocation",
]


@runtime_checkable
class LMMInvocation(Protocol):
    @overload
    async def __call__(
        self,
        *,
        context: Sequence[LMMContextElement],
        tools: Sequence[ToolSpecification] | None = None,
        require_tool: ToolSpecification | bool = False,
        output: Literal["text", "json"] = "text",
        stream: Literal[True],
        **extra: Any,
    ) -> LMMOutputStream: ...

    @overload
    async def __call__(
        self,
        *,
        context: Sequence[LMMContextElement],
        tools: Sequence[ToolSpecification] | None = None,
        require_tool: ToolSpecification | bool = False,
        output: Literal["text", "json"] = "text",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    async def __call__(
        self,
        *,
        context: Sequence[LMMContextElement],
        tools: Sequence[ToolSpecification] | None = None,
        require_tool: ToolSpecification | bool = False,
        output: Literal["text", "json"] = "text",
        stream: bool,
        **extra: Any,
    ) -> LMMOutputStream | LMMOutput: ...

    async def __call__(  # noqa: PLR0913
        self,
        *,
        context: Sequence[LMMContextElement],
        tools: Sequence[ToolSpecification] | None = None,
        require_tool: ToolSpecification | bool = False,
        output: Literal["text", "json"] = "text",
        stream: bool = False,
        **extra: Any,
    ) -> LMMOutputStream | LMMOutput: ...
