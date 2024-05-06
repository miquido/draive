from collections.abc import Callable
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable

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
        tools: Toolbox | None,
        output: Literal["text", "json"],
        stream: Literal[True],
        **extra: Any,
    ) -> LMMCompletionStream: ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None,
        output: Literal["text", "json"],
        stream: Callable[[LMMCompletionStreamingUpdate], None],
        **extra: Any,
    ) -> LMMCompletionMessage: ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None,
        output: Literal["text", "json"],
        stream: Literal[False],
        **extra: Any,
    ) -> LMMCompletionMessage: ...

    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        output: Literal["text", "json"] = "text",
        stream: Callable[[LMMCompletionStreamingUpdate], None] | bool = False,
        **extra: Any,
    ) -> LMMCompletionStream | LMMCompletionMessage: ...
