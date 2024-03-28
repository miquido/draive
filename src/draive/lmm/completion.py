from typing import Literal, Protocol, Self, overload, runtime_checkable

from draive.lmm.message import (
    LMMCompletionMessage,
    LMMCompletionStreamingUpdate,
)
from draive.tools import Toolbox
from draive.types import UpdateSend

__all__ = [
    "LMMCompletion",
    "LMMCompletionStream",
]


class LMMCompletionStream(Protocol):
    def __aiter__(self) -> Self:
        ...

    async def __anext__(self) -> LMMCompletionStreamingUpdate:
        ...


@runtime_checkable
class LMMCompletion(Protocol):
    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        stream: Literal[True],
    ) -> LMMCompletionStream:
        ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        stream: UpdateSend[LMMCompletionStreamingUpdate],
    ) -> LMMCompletionMessage:
        ...

    @overload
    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        output: Literal["text", "json"] = "text",
    ) -> LMMCompletionMessage:
        ...

    async def __call__(
        self,
        *,
        context: list[LMMCompletionMessage],
        tools: Toolbox | None = None,
        output: Literal["text", "json"] = "text",
        stream: UpdateSend[LMMCompletionStreamingUpdate] | bool = False,
    ) -> LMMCompletionStream | LMMCompletionMessage:
        ...
