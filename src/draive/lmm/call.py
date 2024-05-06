from collections.abc import Callable
from typing import Any, Literal, overload

from draive.lmm.completion import LMMCompletionStream
from draive.lmm.message import (
    LMMCompletionMessage,
    LMMCompletionStreamingUpdate,
)
from draive.lmm.state import LMM
from draive.scope import ctx
from draive.tools import Toolbox

__all__ = [
    "lmm_completion",
]


@overload
async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: Literal[True],
    **extra: Any,
) -> LMMCompletionStream: ...


@overload
async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: Callable[[LMMCompletionStreamingUpdate], None],
    **extra: Any,
) -> LMMCompletionMessage: ...


@overload
async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMCompletionMessage: ...


async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: Callable[[LMMCompletionStreamingUpdate], None] | bool = False,
    **extra: Any,
) -> LMMCompletionStream | LMMCompletionMessage:
    match stream:
        case False:
            return await ctx.state(LMM).completion(
                context=context,
                tools=tools,
                output=output,
                stream=False,
                **extra,
            )
        case True:
            return await ctx.state(LMM).completion(
                context=context,
                tools=tools,
                output=output,
                stream=True,
                **extra,
            )
        case progress:
            return await ctx.state(LMM).completion(
                context=context,
                tools=tools,
                output=output,
                stream=progress,
                **extra,
            )
