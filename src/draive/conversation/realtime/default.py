from asyncio import CancelledError
from collections import deque
from collections.abc import Generator, MutableSequence, Sequence
from types import TracebackType
from typing import Any

from haiway import State, ctx

from draive.conversation.realtime.types import (
    RealtimeConversationSession,
    RealtimeConversationSessionScope,
)
from draive.conversation.state import ConversationMemory
from draive.conversation.types import (
    ConversationAssistantTurn,
    ConversationEvent,
    ConversationTurn,
    ConversationUserTurn,
)
from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelOutput,
    ModelReasoning,
    ModelReasoningChunk,
    ModelSession,
    ModelSessionEvent,
    ModelSessionOutputChunk,
    ModelSessionOutputSelection,
    ModelSessionScope,
    ModelToolRequest,
    ModelToolResponse,
    RealtimeGenerativeModel,
)
from draive.multimodal import MultimodalContent, MultimodalContentPart
from draive.tools import Toolbox
from draive.utils import ProcessingEvent

__all__ = ("realtime_conversation_preparing",)


def realtime_conversation_preparing(  # noqa: C901, PLR0915
    *,
    instructions: ModelInstructions,
    toolbox: Toolbox,
    memory: ConversationMemory,
    output: ModelSessionOutputSelection,
    **extra: Any,
) -> RealtimeConversationSessionScope:
    session_scope: ModelSessionScope

    async def open_session() -> RealtimeConversationSession:  # noqa: C901, PLR0915
        nonlocal session_scope
        session_scope = RealtimeGenerativeModel.session(
            instructions=instructions,
            tools=toolbox.model_tools(),
            context=await memory.recall(),
            output=output,
            **extra,
        )
        session: ModelSession = await session_scope.__aenter__()
        tools_output: deque[MultimodalContentPart | ConversationEvent] = deque()

        async def remember_turn(turn: ConversationTurn | None) -> None:
            if turn is None:
                return

            try:
                await memory.remember(turn)

            except BaseException as exc:
                ctx.log_error(
                    "Failed to persist conversation memory context",
                    exception=exc,
                )

        async def handle_tool_request(
            tool_request: ModelToolRequest,
            /,
        ) -> None:
            try:
                ctx.log_debug(f"Requested tool ({tool_request.identifier}) handling...")
                async for chunk in toolbox.handle(tool_request):
                    if isinstance(chunk, ModelToolResponse):
                        tools_output.append(ConversationEvent.tool_response(chunk))
                        # deliver the result directly to input
                        await session._writing(chunk)  # pyright: ignore[reportPrivateUsage]

                    elif isinstance(chunk, ProcessingEvent):
                        tools_output.append(ConversationEvent.tool_event(chunk))

                    else:
                        assert isinstance(chunk, MultimodalContentPart)  # nosec: B101
                        # TODO: we should probably remember it as actual
                        # output to include within context and memory
                        ctx.log_warning("Tool direct outputs are not preserved within context...")
                        tools_output.append(chunk)

            except CancelledError:
                ctx.log_debug(f"...tool request ({tool_request.identifier}) handling cancelled!")
                raise  # reraise cancellation

            except BaseException as exc:
                ctx.log_error(
                    f"...tool request ({tool_request.identifier}) handling failed!",
                    exception=exc,
                )

            else:
                ctx.log_debug(f"...tool request ({tool_request.identifier}) handling completed!")

        pending_elements: MutableSequence[ModelInput | ModelOutput] = []

        async def read() -> MultimodalContentPart | ModelReasoningChunk | ConversationEvent:  # noqa: C901, PLR0912
            while True:
                if tools_output:
                    return tools_output.popleft()

                chunk: ModelSessionOutputChunk = await session._reading()  # pyright: ignore[reportPrivateUsage]
                if isinstance(chunk, ModelReasoningChunk):
                    return chunk

                elif isinstance(chunk, ModelToolRequest):
                    ctx.spawn(handle_tool_request, chunk)
                    return ConversationEvent.tool_request(chunk)

                elif isinstance(chunk, ModelSessionEvent):
                    if chunk.event == "turn_completed":
                        if isinstance(chunk.content, ModelInput):
                            if chunk.content.contains_tools:
                                pending_elements.append(chunk.content)

                            else:
                                await remember_turn(_user_turn(chunk.content))

                        elif isinstance(chunk.content, ModelOutput):
                            if chunk.content.contains_tools:
                                pending_elements.append(chunk.content)

                            else:
                                await remember_turn(
                                    _assistant_turn(
                                        chunk.content,
                                        *pending_elements,
                                    )
                                )
                                pending_elements.clear()

                        elif pending_elements:
                            await remember_turn(_assistant_turn(*pending_elements))
                            pending_elements.clear()

                        else:
                            continue  # skip

                    elif isinstance(chunk.content, State | None):
                        return ConversationEvent.of(
                            chunk.event,
                            content=chunk.content,
                            meta=chunk.meta,
                        )

                    else:
                        ctx.log_error(f"Received unsupported session event: {chunk.event}")

                else:
                    return chunk

        async def write(
            input: MultimodalContentPart | ConversationEvent,  # noqa: A002
        ) -> None:
            if isinstance(input, ConversationEvent):
                event: ModelSessionEvent
                if input.event == "context_updated":
                    event = ModelSessionEvent.context_updated(
                        await memory.recall(),
                        meta=input.meta,
                    )

                elif input.content is None:
                    event = ModelSessionEvent.of(
                        input.event,
                        meta=input.meta,
                    )

                else:
                    event = ModelSessionEvent.of(
                        input.event,
                        content=input.content,
                        meta=input.meta,
                    )

                await session._writing(event)  # pyright: ignore[reportPrivateUsage]

            else:
                await session._writing(input)  # pyright: ignore[reportPrivateUsage]

        return RealtimeConversationSession(
            reading=read,
            writing=write,
        )

    async def close_session(
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await session_scope.__aexit__(  # noqa: F821
            exc_type,
            exc_val,
            exc_tb,
        )

    return RealtimeConversationSessionScope(
        opening=open_session,
        closing=close_session,
    )


def _user_turn_content(element: ModelInput) -> Generator[MultimodalContent]:
    for block in element.input:
        if isinstance(block, MultimodalContent):
            yield block


def _user_turn(
    element: ModelInput,
) -> ConversationUserTurn | None:
    return ConversationUserTurn.of(*_user_turn_content(element))


def _assistant_turn_content(
    elements: Sequence[ModelInput | ModelOutput],
) -> Generator[MultimodalContent | ModelReasoning | ConversationEvent]:
    for element in elements:
        if isinstance(element, ModelInput):
            for block in element.input:
                if isinstance(block, MultimodalContent):
                    yield block

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            for block in element.output:
                if isinstance(block, MultimodalContent | ModelReasoning | ConversationEvent):
                    yield block


def _assistant_turn(
    *elements: ModelInput | ModelOutput,
) -> ConversationAssistantTurn | None:
    return ConversationAssistantTurn.of(*_assistant_turn_content(elements))
