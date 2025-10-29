from asyncio import CancelledError
from types import TracebackType
from typing import Any

from haiway import ctx

from draive.conversation.realtime.types import (
    RealtimeConversationSession,
    RealtimeConversationSessionScope,
)
from draive.conversation.types import (
    ConversationEvent,
    ConversationInputChunk,
    ConversationOutputChunk,
)
from draive.models import (
    ModelInstructions,
    ModelMemory,
    ModelSession,
    ModelSessionOutputSelection,
    ModelSessionScope,
    RealtimeGenerativeModel,
    Toolbox,
)
from draive.models.types import (
    ModelInputChunk,
    ModelOutputChunk,
    ModelSessionEvent,
    ModelToolRequest,
    ModelToolResponse,
)

__all__ = ("realtime_conversation_preparing",)


async def realtime_conversation_preparing(  # noqa: C901
    *,
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ModelMemory,
    output: ModelSessionOutputSelection,
    **extra: Any,
) -> RealtimeConversationSessionScope:
    """Default realtime conversation session preparation.

    Opens a realtime generative model session, adapts session I/O to conversation-level
    chunks, and wires tool requests handling using the provided toolbox.
    """
    # TODO: rework RealtimeLMM interface to allow instruction and tools changes during the session
    session_scope: ModelSessionScope = await RealtimeGenerativeModel.session(
        instructions=instructions,
        memory=memory,
        tools=toolbox.available_tools_declaration(),
        output=output,
        **extra,
    )

    async def open_session() -> RealtimeConversationSession:  # noqa: C901
        session: ModelSession = await session_scope.__aenter__()

        async def handle_tool_request(
            tool_request: ModelToolRequest,
            /,
        ) -> None:
            try:
                ctx.log_debug(
                    f"Handling tool ({tool_request.tool}) request ({tool_request.identifier}) ..."
                )
                response: ModelToolResponse = await toolbox.respond(tool_request)
                if response.handling in ("response", "output", "output_extension"):
                    ctx.log_warning(
                        f"Tool handling `{response.handling}` is not supported in"
                        " realtime conversation, using regular result handling instead"
                    )

                # check if task was not cancelled before passing the result to input
                ctx.check_cancellation()
                # deliver the result directly to input
                await session.writing(response)

            except CancelledError:
                ctx.log_debug(
                    f"...tool request ({tool_request.identifier}) handling cancelled!",
                )

            except BaseException as exc:
                ctx.log_error(
                    f"...tool request ({tool_request.identifier}) handling failed!",
                    exception=exc,
                )

            else:
                ctx.log_debug(
                    f"...tool request ({tool_request.identifier}) handling completed!",
                )

        async def read() -> ConversationOutputChunk | ConversationEvent:
            while True:
                match await session.reading():
                    case ModelOutputChunk() as chunk:
                        ctx.log_debug("...received response chunk...")
                        return ConversationOutputChunk.of(
                            chunk.content,
                            eod=chunk.eod,
                        )

                    case ModelSessionEvent() as event:
                        ctx.log_debug(f"...received {event.category} event...")
                        return ConversationEvent.of(
                            event.category,
                            content=event.content,
                            meta=event.meta,
                        )

                    case ModelToolRequest() as tool_request:
                        ctx.log_debug(f"...received {tool_request.tool} request...")
                        ctx.spawn(
                            handle_tool_request,
                            tool_request,
                        )

        async def write(
            input: ConversationInputChunk | ConversationEvent,  # noqa: A002
        ) -> None:
            if isinstance(input, ConversationEvent):
                await session.writing(
                    input=ModelSessionEvent.of(
                        input.category,
                        content=input.content,
                    )
                )

            else:
                await session.writing(
                    input=ModelInputChunk.of(
                        input.content,
                        eod=input.eod,
                    )
                )

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
