from collections.abc import AsyncIterator
from typing import Any, Literal, final, overload

from haiway import State, ctx

from draive.lmm.types import (
    LMMCompleting,
    LMMContext,
    LMMInstruction,
    LMMMemory,
    LMMOutput,
    LMMOutputSelection,
    LMMSessionOutputSelection,
    LMMSessionPreparing,
    LMMSessionScope,
    LMMStreamOutput,
    LMMTools,
)

__all__ = (
    "LMM",
    "RealtimeLMM",
)


@final
class LMM(State):
    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instruction: LMMInstruction | None = None,
        context: LMMContext,
        tools: LMMTools | None = None,
        output: LMMOutputSelection = "auto",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> LMMOutput: ...

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instruction: LMMInstruction | None = None,
        context: LMMContext,
        tools: LMMTools | None = None,
        output: LMMOutputSelection = "auto",
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput]: ...

    @classmethod
    async def completion(
        cls,
        *,
        instruction: LMMInstruction | None = None,
        context: LMMContext,
        tools: LMMTools | None = None,
        output: LMMOutputSelection = "auto",
        stream: bool = False,
        **extra: Any,
    ) -> AsyncIterator[LMMStreamOutput] | LMMOutput:
        if stream:
            return await ctx.state(cls).completing(
                instruction=instruction,
                context=context,
                tools=tools,
                output=output,
                stream=True,
                **extra,
            )

        else:
            return await ctx.state(cls).completing(
                instruction=instruction,
                context=context,
                tools=tools,
                output=output,
                **extra,
            )

    completing: LMMCompleting


@final
class RealtimeLMM(State):
    @classmethod
    async def session(
        cls,
        *,
        instruction: LMMInstruction | None = None,
        memory: LMMMemory | None = None,
        tools: LMMTools | None = None,
        output: LMMSessionOutputSelection = "auto",
        **extra: Any,
    ) -> LMMSessionScope:
        return await ctx.state(cls).session_preparing(
            instruction=instruction,
            memory=memory,
            output=output,
            tools=tools,
            **extra,
        )

    session_preparing: LMMSessionPreparing
