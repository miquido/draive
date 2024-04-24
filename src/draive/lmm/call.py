from collections.abc import Callable
from typing import Literal, overload

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
) -> LMMCompletionStream: ...


@overload
async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    stream: Callable[[LMMCompletionStreamingUpdate], None],
) -> LMMCompletionMessage: ...


@overload
async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
) -> LMMCompletionMessage: ...


async def lmm_completion(
    *,
    context: list[LMMCompletionMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: Callable[[LMMCompletionStreamingUpdate], None] | bool = False,
) -> LMMCompletionStream | LMMCompletionMessage:
    match stream:
        case False:
            return await ctx.state(LMM).completion(
                context=context,
                tools=tools,
                output=output,
            )
        case True:
            return await ctx.state(LMM).completion(
                context=context,
                tools=tools,
                stream=True,
            )
        case progress:
            return await ctx.state(LMM).completion(
                context=context,
                tools=tools,
                stream=progress,
            )
