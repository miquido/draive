from collections.abc import AsyncGenerator, AsyncIterator, Iterable

from haiway import State, ctx

from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.realtime.default import realtime_process
from draive.realtime.types import RealtimeOutputSelection, RealtimeProcessing
from draive.tools import Tool, Toolbox

__all__ = ("Realtime",)


class Realtime(State):
    @classmethod
    async def process(
        cls,
        *,
        instruction: Instruction | str | None = None,
        input: AsyncIterator[Multimodal],  # noqa: A002
        tools: Toolbox | Iterable[Tool] | None = None,
        output: RealtimeOutputSelection = "auto",
    ) -> AsyncIterator[MultimodalContent]:
        async def input_stream() -> AsyncGenerator[MultimodalContent]:
            async for element in input:
                yield MultimodalContent.of(element)

        return await ctx.state(cls).processing(
            instruction=Instruction.of(instruction),
            input_stream=input_stream(),
            toolbox=Toolbox.of(tools),
            output=output,
        )

    processing: RealtimeProcessing = realtime_process
