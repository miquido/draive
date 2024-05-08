from collections.abc import Callable
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable

from draive.lmm.message import (
    LMMMessage,
    LMMStreamingUpdate,
)
from draive.tools import Toolbox

__all__ = [
    "LMMCompletion",
    "LMMCompletionStream",
]


class LMMCompletionStream(Protocol):
    def __aiter__(self) -> Self: ...

    async def __anext__(self) -> LMMStreamingUpdate: ...


@runtime_checkable
class LMMCompletion(Protocol):
    @overload
    async def __call__(
        self,
        *,
        context: list[LMMMessage],
        tools: Toolbox | None,
        output: Literal["text", "json"],
        stream: Literal[True],
        **extra: Any,
    ) -> LMMCompletionStream: ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMMessage],
        tools: Toolbox | None,
        output: Literal["text", "json"],
        stream: Callable[[LMMStreamingUpdate], None],
        **extra: Any,
    ) -> LMMMessage: ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMMessage],
        tools: Toolbox | None,
        output: Literal["text", "json"],
        stream: Literal[False],
        **extra: Any,
    ) -> LMMMessage: ...

    async def __call__(
        self,
        *,
        context: list[LMMMessage],
        tools: Toolbox | None = None,
        output: Literal["text", "json"] = "text",
        stream: Callable[[LMMStreamingUpdate], None] | bool = False,
        **extra: Any,
    ) -> LMMCompletionStream | LMMMessage: ...
