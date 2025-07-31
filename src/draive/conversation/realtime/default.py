from asyncio import CancelledError
from collections.abc import Sequence
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
    ConversationMemory,
    ConversationMessage,
    ConversationMessageChunk,
    ConversationStreamElement,
)
from draive.instructions import Instruction
from draive.lmm import (
    LMMContext,
    LMMContextElement,
    LMMMemory,
    LMMSession,
    LMMSessionEvent,
    LMMSessionScope,
    LMMStreamChunk,
    LMMToolRequest,
    LMMToolResponse,
    RealtimeLMM,
)
from draive.lmm.types import LMMCompletion, LMMInput
from draive.tools import Toolbox
from draive.utils import MEMORY_NONE, Memory, MemoryRecalling, MemoryRemembering

__all__ = ("realtime_conversation_preparing",)


async def realtime_conversation_preparing(  # noqa: C901
    *,
    instruction: Instruction | None,
    memory: ConversationMemory | None,
    toolbox: Toolbox,
    **extra: Any,
) -> RealtimeConversationSessionScope:
    session_memory: LMMMemory
    match memory:
        case None:
            session_memory = MEMORY_NONE

        case Memory():
            session_memory = Memory(
                recall=_to_lmm_recall(memory.recall),
                remember=_to_lmm_remember(memory.remember),
            )

    # TODO: rework RealtimeLMM interface to allow instruction and tools changes during the session
    session_scope: LMMSessionScope = await RealtimeLMM.session(
        instruction=Instruction.formatted(instruction),
        memory=session_memory,
        tools=toolbox.available_tools(),
        output="auto",
    )

    async def open_session() -> RealtimeConversationSession:  # noqa: C901
        session: LMMSession = await session_scope.__aenter__()

        async def handle_tool_request(
            tool_request: LMMToolRequest,
            /,
        ) -> None:
            try:
                ctx.log_debug(
                    f"Handling tool ({tool_request.tool}) request ({tool_request.identifier}) ...",
                )
                response: LMMToolResponse = await toolbox.respond(tool_request)
                if response.handling in ("completion", "extension"):
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

        async def read() -> ConversationStreamElement:
            while True:
                match await session.reading():
                    case LMMStreamChunk() as chunk:
                        ctx.log_debug("...received response chunk...")
                        return ConversationMessageChunk.model(
                            chunk.content,
                            identifier=chunk.meta.identifier,
                            message_identifier=chunk.meta.get_uuid(
                                "message_identifier",
                                default=uuid4(),
                            ),
                            eod=chunk.eod,
                        )

                    case LMMSessionEvent() as event:
                        ctx.log_debug(f"...received {event.category} event...")
                        return ConversationEvent.of(
                            event.category,
                            meta=event.meta,
                        )

                    case LMMToolRequest() as tool_request:
                        ctx.log_debug(f"...received {tool_request.tool} request...")
                        ctx.spawn(
                            handle_tool_request,
                            tool_request,
                        )
                        return ConversationEvent.of(
                            category="tool.call",
                            meta={
                                "call_id": tool_request.identifier,
                                "tool": tool_request.tool,
                            },
                        )

        async def write(
            input: ConversationStreamElement,  # noqa: A002
        ) -> None:
            if isinstance(input, ConversationMessageChunk):
                # pass chunk to LMM
                await session.writing(
                    input=LMMStreamChunk.of(
                        input.content,
                        eod=input.eod,
                        meta=input.meta,
                    )
                )

            else:
                assert isinstance(input, ConversationEvent)  # nosec: B101
                # pass event to LMM
                await session.writing(
                    input=LMMSessionEvent.of(
                        input.category,
                        meta=input.meta,
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


def _to_lmm_recall(
    recall: MemoryRecalling[Sequence[ConversationMessage]],
    /,
) -> MemoryRecalling[LMMContext]:
    async def lmm_recall(
        **extra: Any,
    ) -> LMMContext:
        return tuple(element.to_lmm_context() for element in await recall())

    return lmm_recall


def _to_lmm_remember(
    remember: MemoryRemembering[ConversationMessage],
    /,
) -> MemoryRemembering[LMMContextElement]:
    async def lmm_remember(
        *items: LMMContextElement,
        **extra: Any,
    ) -> None:
        await remember(
            *(
                ConversationMessage.from_lmm_context(element)
                for element in items
                if isinstance(element, LMMInput | LMMCompletion)
            )
        )

    return lmm_remember
