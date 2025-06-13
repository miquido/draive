from asyncio import CancelledError, Task
from types import TracebackType
from typing import Any
from uuid import uuid4

from haiway import ctx

from draive.conversation.realtime.types import (
    RealtimeConversationSession,
    RealtimeConversationSessionScope,
)
from draive.conversation.types import (
    ConversationEvent,
    ConversationMessageChunk,
    ConversationStreamElement,
)
from draive.instructions import Instruction
from draive.lmm import (
    LMMMemory,
    LMMSession,
    LMMSessionEvent,
    LMMSessionScope,
    LMMStreamChunk,
    LMMToolRequest,
    LMMToolResponse,
    RealtimeLMM,
)
from draive.multimodal import MultimodalContent
from draive.tools import Toolbox

__all__ = ("realtime_conversation_preparing",)


async def realtime_conversation_preparing(  # noqa: C901
    *,
    instruction: Instruction | None,
    memory: LMMMemory | None,
    toolbox: Toolbox,
    **extra: Any,
) -> RealtimeConversationSessionScope:
    session_memory: LMMMemory
    match memory:
        case None:
            session_memory = LMMMemory.none

        case LMMMemory():
            session_memory = memory

        case context:
            session_memory = LMMMemory.constant(context)

    session_scope: LMMSessionScope = await RealtimeLMM.session(
        instruction=Instruction.formatted(instruction),
        memory=session_memory,
        tools=toolbox.current_tools(),
        output="auto",
    )

    async def open_session() -> RealtimeConversationSession:  # noqa: C901
        session: LMMSession = await session_scope.__aenter__()
        pending_tool_requests: dict[str, Task[None]] = {}

        async def handle_tool_request(
            tool_request: LMMToolRequest,
            /,
        ) -> None:
            try:
                ctx.log_debug(f"Handling tool request ({tool_request.identifier})...")
                response: LMMToolResponse = await toolbox.respond(tool_request)
                # check if task was not cancelled before passing the result to input
                ctx.check_cancellation()
                # deliver the result directly to input
                await session.write(response)

            except CancelledError:
                pass  # just cancelled, ignore

            finally:
                # update pending tools - we have delivered or ignored the result
                del pending_tool_requests[tool_request.identifier]
                ctx.log_debug(f"...tool request ({tool_request.identifier}) handling finished!")

        async def read() -> ConversationStreamElement:
            while True:
                match await session.reading():
                    case LMMStreamChunk() as chunk:
                        return ConversationMessageChunk.model(
                            chunk.content,
                            message_identifier=chunk.meta.origin_identifier or uuid4(),
                            eod=chunk.eod,
                        )

                    case LMMSessionEvent() as event:
                        match event.category:
                            case "output.completed":
                                return ConversationMessageChunk.model(
                                    MultimodalContent.empty,
                                    message_identifier=event.meta.origin_identifier or uuid4(),
                                    eod=True,  # ensure sending eod
                                )

                            case "input.started":
                                return ConversationEvent.of(
                                    "interrupted",
                                    meta=event.meta,
                                )

                            case _:
                                pass  # temporary ignore of other events

                        return ConversationEvent.of(
                            event.category,
                            meta=event.meta,
                        )

                    case LMMToolRequest() as tool_request:
                        pending_tool_requests[tool_request.identifier] = ctx.spawn(
                            handle_tool_request,
                            tool_request,
                        )
                        continue

        async def write(
            input: ConversationStreamElement,  # noqa: A002
        ) -> None:
            if isinstance(input, ConversationEvent):
                return  # events do not propagate to LMM

            chunk: LMMStreamChunk = LMMStreamChunk.of(
                input.content,
                eod=input.eod,
                meta=input.meta,
            )
            # pass chunk to LMM
            await session.writing(input=chunk)

        return RealtimeConversationSession(
            reading=read,
            writing=write,
        )

    async def close_session(
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await session_scope.__aexit__(
            exc_type,
            exc_val,
            exc_tb,
        )

    return RealtimeConversationSessionScope(
        opening=open_session,
        closing=close_session,
    )
