from asyncio import CancelledError, Task
from collections.abc import AsyncIterator, Sequence
from typing import Any

from haiway import AsyncQueue, ctx

from draive.conversation.types import (
    ConversationEvent,
    ConversationMessage,
    ConversationMessageChunk,
    ConversationStreamElement,
    RealtimeConversationMemory,
)
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
from draive.lmm.types import LMMContextElement
from draive.multimodal import MetaContent
from draive.tools import Toolbox
from draive.utils import Processing, ProcessingEvent

__all__ = ("realtime_conversation",)


async def realtime_conversation(  # noqa: C901, PLR0915
    *,
    instruction: Instruction | None,
    input_stream: AsyncIterator[ConversationStreamElement],
    memory: RealtimeConversationMemory,
    toolbox: Toolbox,
    **extra: Any,
) -> AsyncIterator[ConversationStreamElement]:
    initial_context: Sequence[LMMContextElement] = tuple(
        message.as_lmm_context_element()
        for message in await memory.recall()
        if isinstance(message, ConversationMessage)  # skip events in LMM context
    )

    input_queue = AsyncQueue[LMMStreamInput]()

    async def process_input() -> None:
        try:
            # TODO: include guardrails check
            async for element in input_stream:
                ctx.spawn(memory.remember, element)  # add to memory without waiting

                if isinstance(element, ConversationEvent):
                    continue  # skip events in LMM context

                # we are expecting continuous stream of data without turns,
                # this api will work only with LMMSession implementations
                # which can automatically detect turn ends if needed
                input_queue.enqueue(
                    LMMStreamChunk.of(
                        element.content,
                        eod=element.eod,
                    )
                )

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
            initial_context=initial_context,
            tools=toolbox.available_tools(),
        )
        del initial_context

    except BaseException as exc:
        input_processing_task.cancel()
        raise exc

    output_queue = AsyncQueue[ConversationStreamElement]()

    async def report_event(
        event: ProcessingEvent,
        /,
    ) -> None:
        output_queue.enqueue(
            ConversationEvent.of(
                event.name,
                identifier=event.identifier,
                content=event.content,
                meta=event.meta,
            )
        )

    with ctx.updated(ctx.state(Processing).updated(event_reporting=report_event)):

        async def process_output() -> None:
            try:
                async for chunk in session_output:
                    match chunk:
                        case LMMStreamChunk() as chunk:
                            output_chunk: ConversationMessageChunk = ConversationMessageChunk.model(
                                chunk.content,
                                eod=chunk.eod,
                            )
                            ctx.spawn(
                                memory.remember, output_chunk
                            )  # add to memory without waiting
                            output_queue.enqueue(output_chunk)

                        case LMMSessionEvent() as event:
                            conversation_event: ConversationEvent
                            match event.category:
                                case "interrupted":
                                    # cancel pending tools on interrupt
                                    for task in pending_tool_requests.values():
                                        task.cancel()

                                    conversation_event = ConversationEvent.of(
                                        "interrupted",
                                        meta=event.meta,
                                    )

                                case "completed":
                                    assert not pending_tool_requests  # nosec: B101
                                    conversation_event = ConversationEvent.of(
                                        "completed",
                                        meta=event.meta,
                                    )

                                case other:
                                    conversation_event = ConversationEvent.of(
                                        other,
                                        content=MetaContent.of(other),
                                        meta=event.meta,
                                    )

                            ctx.spawn(
                                memory.remember,
                                conversation_event,
                            )  # add to memory without waiting
                            output_queue.enqueue(conversation_event)

                        case LMMToolRequest() as tool_request:
                            pending_tool_requests[tool_request.identifier] = ctx.spawn(
                                handle_tool_request,
                                tool_request,
                            )

            finally:
                input_processing_task.cancel()

        ctx.spawn(process_output)

    return output_queue
