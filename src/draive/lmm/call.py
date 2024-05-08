from collections.abc import Callable
from typing import Any, Literal, overload

from draive.lmm.completion import LMMCompletionStream
from draive.lmm.message import (
    LMMMessage,
    LMMStreamingUpdate,
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
    context: list[LMMMessage],
    tools: Toolbox | None = None,
    stream: Literal[True],
    **extra: Any,
) -> LMMCompletionStream: ...


@overload
async def lmm_completion(
    *,
    context: list[LMMMessage],
    tools: Toolbox | None = None,
    stream: Callable[[LMMStreamingUpdate], None],
    **extra: Any,
) -> LMMMessage: ...


@overload
async def lmm_completion(
    *,
    context: list[LMMMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: Literal[False] = False,
    **extra: Any,
) -> LMMMessage: ...


async def lmm_completion(
    *,
    context: list[LMMMessage],
    tools: Toolbox | None = None,
    output: Literal["text", "json"] = "text",
    stream: Callable[[LMMStreamingUpdate], None] | bool = False,
    **extra: Any,
) -> LMMCompletionStream | LMMMessage:
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
