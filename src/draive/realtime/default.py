from asyncio import CancelledError, Task
from collections.abc import AsyncGenerator, AsyncIterator
from typing import (
    Any,
)

from haiway import AsyncQueue, ctx

from draive.instructions import Instruction
from draive.lmm import (
    LMMSession,
    LMMSessionEvent,
    LMMSessionOutput,
    LMMStreamChunk,
    LMMStreamInput,
    LMMToolRequest,
    LMMToolResponse,
)
from draive.multimodal import MetaContent, MultimodalContent
from draive.realtime.types import RealtimeOutputSelection
from draive.tools import Toolbox

__all__ = ("realtime_process",)


async def realtime_process(  # noqa: C901
    *,
    instruction: Instruction | None,
    input_stream: AsyncIterator[MultimodalContent],
    toolbox: Toolbox,
    output: RealtimeOutputSelection,
    **extra: Any,
) -> AsyncIterator[MultimodalContent]:
    input_queue = AsyncQueue[LMMStreamInput]()

    async def process_input() -> None:
        try:
            async for element in input_stream:
                # we are expecting continuous stream of data without turns,
                # this api will work only with LMMSession implementations
                # which can automatically detect turn ends if needed
                input_queue.enqueue(LMMStreamChunk.of(element))

            input_queue.finish()

        except CancelledError:
            # just finish input on cancel
            input_queue.finish()

        except BaseException as exc:
            input_queue.finish(exception=exc)

    pending_tool_requests: dict[str, Task[None]] = {}

    async def handle_tool_request(
        tool_request: LMMToolRequest,
        /,
    ) -> None:
        try:
            response: LMMToolResponse = await toolbox.respond(tool_request)
            # check if task was not cancelled before passing the result to input
            ctx.check_cancellation()
            # deliver the result directly to input
            input_queue.enqueue(response)

        except CancelledError:
            pass  # just cancelled, ignore

        finally:
            # update pending tools - we have delivered or ignored the result
            del pending_tool_requests[tool_request.identifier]

    input_processing_task: Task[None] = ctx.spawn(process_input)

    session_output: AsyncIterator[LMMSessionOutput]
    try:
        session_output = await LMMSession.prepare(
            instruction=instruction,
            input_stream=input_queue,
            output=output,
            tools=toolbox.available_tools(),
        )

    except BaseException as exc:
        input_processing_task.cancel()
        raise exc

    async def process_output() -> AsyncGenerator[MultimodalContent]:
        try:
            async for chunk in session_output:
                match chunk:
                    case LMMStreamChunk() as chunk:
                        yield chunk.content

                    case LMMSessionEvent() as event:
                        match event.category:
                            case "interrupted":
                                # cancel pending tools on interrupt
                                for task in pending_tool_requests.values():
                                    task.cancel()

                                yield MultimodalContent.of(
                                    MetaContent.of(
                                        "interrupted",
                                        meta=event.meta,
                                    )
                                )

                            case "completed":
                                assert not pending_tool_requests  # nosec: B101
                                yield MultimodalContent.of(
                                    MetaContent.of(
                                        "completed",
                                        meta=event.meta,
                                    )
                                )

                            case other:
                                yield MultimodalContent.of(
                                    MetaContent.of(
                                        other,
                                        meta=event.meta,
                                    )
                                )

                    case LMMToolRequest() as tool_request:
                        pending_tool_requests[tool_request.identifier] = ctx.spawn(
                            handle_tool_request,
                            tool_request,
                        )

        finally:
            input_processing_task.cancel()

    return ctx.stream(process_output)
