from collections.abc import Callable
from typing import Literal, Protocol, Self, overload, runtime_checkable

from draive.lmm.message import (
    LMMCompletionMessage,
    LMMCompletionStreamingUpdate,
)
from draive.tools import Toolbox

__all__ = [
    "LMMCompletion",
    "LMMCompletionStream",
]


class LMMCompletionStream(Protocol):
    def __aiter__(self) -> Self: ...

    async def __anext__(self) -> LMMCompletionStreamingUpdate: ...


@runtime_checkable
class LMMCompletion(Protocol):
    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        stream: Literal[True],
    ) -> LMMCompletionStream: ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        stream: Callable[[LMMCompletionStreamingUpdate], None],
    ) -> LMMCompletionMessage: ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        output: Literal["text", "json"] = "text",
    ) -> LMMCompletionMessage: ...

    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        output: Literal["text", "json"] = "text",
        stream: Callable[[LMMCompletionStreamingUpdate], None] | bool = False,
    ) -> LMMCompletionStream | LMMCompletionMessage: ...
