import asyncio
from asyncio import CancelledError
from collections import deque
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Mapping,
    MutableSequence,
    Sequence,
)
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
    pending_reading: asyncio.Task[ModelSessionOutputChunk] | None = None
    flushing_pending: Callable[[], Awaitable[None]] | None = None

    async def open_session() -> RealtimeConversationSession:  # noqa: C901, PLR0915
        nonlocal session_scope
        nonlocal flushing_pending
        session_scope = RealtimeGenerativeModel.session(
            instructions=instructions,
            tools=toolbox.model_tools(iteration=0),
            context=await memory.recall(),
            output=output,
            **extra,
        )
        session: ModelSession = await session_scope.__aenter__()
        tools_output: deque[MultimodalContentPart | ConversationEvent] = deque()
        tools_output_available: asyncio.Event = asyncio.Event()
        tools_running: set[asyncio.Task[None]] = set()

        def emit_tool_output(output: MultimodalContentPart | ConversationEvent, /) -> None:
            tools_output.append(output)
            tools_output_available.set()

        async def remember_turn(turn: ConversationTurn | None) -> None:
            if not turn:
                return  # nothing to remember, an empty turn would add no context

            try:
                await memory.remember(turn)

            except Exception as exc:  # cancellation must propagate
                ctx.log_error(
                    "Failed to persist conversation memory context",
                    exception=exc,
                )

        async def handle_tool_request(  # noqa: C901
            tool_request: ModelToolRequest,
            /,
        ) -> None:
            handling_stream: AsyncGenerator[
                ModelToolResponse | ProcessingEvent | MultimodalContentPart
            ] = toolbox.handle((tool_request,))
            answered: bool = False

            async def answer_with_error(message: str, /) -> None:
                # provider sessions wait for every requested tool to be answered
                response: ModelToolResponse = ModelToolResponse(
                    identifier=tool_request.identifier,
                    tool=tool_request.tool,
                    status="error",
                    content=MultimodalContent.of(f"<error>{message}</error>"),
                )
                try:
                    handled_responses[response.identifier] = response
                    emit_tool_output(ConversationEvent.tool_response(response))
                    await session._writing(response)  # pyright: ignore[reportPrivateUsage]

                except Exception as exc:
                    ctx.log_error(
                        f"...tool request ({tool_request.identifier}) error response failed!",
                        exception=exc,
                    )

            try:
                ctx.log_debug(f"Requested tool ({tool_request.identifier}) handling...")
                async for chunk in handling_stream:
                    if isinstance(chunk, ModelToolResponse):
                        handled_responses[chunk.identifier] = chunk
                        emit_tool_output(ConversationEvent.tool_response(chunk))
                        # deliver the result directly to input
                        await session._writing(chunk)  # pyright: ignore[reportPrivateUsage]
                        answered = True

                    elif isinstance(chunk, ProcessingEvent):
                        emit_tool_output(ConversationEvent.tool_event(chunk))

                    else:
                        assert isinstance(chunk, MultimodalContentPart)  # nosec: B101
                        # TODO: we should probably remember it as actual
                        # output to include within context and memory
                        ctx.log_warning("Tool direct outputs are not preserved within context...")
                        emit_tool_output(chunk)

            except CancelledError:
                ctx.log_debug(f"...tool request ({tool_request.identifier}) handling cancelled!")
                if not answered:
                    await answer_with_error("Tool execution was cancelled")

                raise  # reraise cancellation

            except BaseException as exc:
                ctx.log_error(
                    f"...tool request ({tool_request.identifier}) handling failed!",
                    exception=exc,
                )
                if not answered:
                    await answer_with_error("Tool execution failed due to an error")

            else:
                ctx.log_debug(f"...tool request ({tool_request.identifier}) handling completed!")
                if not answered:  # handling ended without any response
                    await answer_with_error("Tool execution failed due to an error")

            finally:
                await handling_stream.aclose()  # release the tools handling stream

        pending_elements: MutableSequence[ModelInput | ModelOutput] = []
        handled_responses: dict[str, ModelToolResponse] = {}

        # tool requests observed as stream chunks, a provider may not report them within
        # its completed turns (or report them only after they were handled), each one is
        # kept as a pending element until a completed turn reports it
        observed_requests: dict[str, ModelOutput] = {}

        def assistant_turn() -> ConversationAssistantTurn | None:
            turn: ConversationAssistantTurn | None = _assistant_turn(
                tuple(pending_elements),
                handled_responses,
            )
            for element in pending_elements:
                if isinstance(element, ModelOutput):
                    for request in element.tool_requests:
                        handled_responses.pop(request.identifier, None)

            pending_elements.clear()
            observed_requests.clear()
            return turn

        async def flush_pending() -> None:
            if pending_elements:
                await remember_turn(assistant_turn())

        flushing_pending = flush_pending

        def pending_exchange_answered() -> bool:
            # a completed exchange has a response delivered for every pending tool request
            requests: set[str] = {
                request.identifier
                for element in pending_elements
                if isinstance(element, ModelOutput)
                for request in element.tool_requests
            }
            return bool(requests) and requests.issubset(handled_responses.keys())

        async def read() -> MultimodalContentPart | ModelReasoningChunk | ConversationEvent:  # noqa: C901, PLR0912, PLR0915
            nonlocal pending_reading
            while True:
                if tools_output:
                    return tools_output.popleft()

                # wait for either the provider or the tools output, whichever comes first
                tools_output_available.clear()
                if pending_reading is None:
                    pending_reading = ctx.spawn(session._reading)  # pyright: ignore[reportPrivateUsage]

                if tools_running:
                    tools_waiting: asyncio.Task[bool] = ctx.spawn(tools_output_available.wait)
                    try:
                        await asyncio.wait(
                            (pending_reading, tools_waiting),
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                    finally:
                        tools_waiting.cancel()

                else:  # nothing can produce tools output, wait for the provider only
                    await asyncio.wait((pending_reading,))

                if not pending_reading.done():
                    continue  # only tools output arrived, provider read stays pending

                # a read cancelled together with the caller stays pending for the next call
                reading_result: asyncio.Task[ModelSessionOutputChunk] = pending_reading
                pending_reading = None
                chunk: ModelSessionOutputChunk = reading_result.result()
                if isinstance(chunk, ModelReasoningChunk):
                    return chunk

                elif isinstance(chunk, ModelToolRequest):
                    handling: asyncio.Task[None] = ctx.spawn(handle_tool_request, chunk)
                    tools_running.add(handling)
                    handling.add_done_callback(tools_running.discard)
                    if chunk.identifier not in observed_requests:
                        observed: ModelOutput = ModelOutput.of(chunk)
                        observed_requests[chunk.identifier] = observed
                        pending_elements.append(observed)

                    return ConversationEvent.tool_request(chunk)

                elif isinstance(chunk, ModelSessionEvent):
                    if chunk.event == "turn_completed":
                        if isinstance(chunk.content, ModelInput):
                            if chunk.content.contains_tools:
                                pending_elements.append(chunk.content)

                            else:
                                if pending_elements:
                                    # the assistant turn was interrupted by the user,
                                    # it ends here instead of attaching to the next one
                                    await remember_turn(assistant_turn())

                                await remember_turn(_user_turn(chunk.content))

                        elif isinstance(chunk.content, ModelOutput):
                            if chunk.content.contains_tools:
                                # the completed output supersedes requests observed as chunks
                                for request in chunk.content.tool_requests:
                                    superseded: ModelOutput | None = observed_requests.pop(
                                        request.identifier,
                                        None,
                                    )
                                    if superseded is not None:
                                        pending_elements[:] = [
                                            element
                                            for element in pending_elements
                                            if element is not superseded
                                        ]

                                pending_elements.append(chunk.content)
                                if pending_exchange_answered():
                                    # the provider closed the turn after the tools were answered,
                                    # the output carries the final answer of the exchange
                                    await remember_turn(assistant_turn())

                            else:
                                pending_elements.append(chunk.content)
                                await remember_turn(assistant_turn())

                        elif pending_elements:
                            await remember_turn(assistant_turn())

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
        if pending_reading is not None:
            pending_reading.cancel()

        if flushing_pending is not None:
            # whatever the assistant produced until now belongs to the memory
            await flushing_pending()

        await session_scope.__aexit__(
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


def _assistant_turn_content(  # noqa: C901
    elements: Sequence[ModelInput | ModelOutput],
    handled_responses: Mapping[str, ModelToolResponse],
) -> Generator[MultimodalContent | ModelReasoning | ConversationEvent]:
    # tool exchanges are remembered as events, the same way as the completion loop does it,
    # requests without a response (and responses without a request) can't be replayed
    # to the model so they are not remembered
    responded: set[str] = {
        response.identifier
        for element in elements
        if isinstance(element, ModelInput)
        for response in element.tool_responses
    }
    requested: set[str] = {
        request.identifier
        for element in elements
        if isinstance(element, ModelOutput)
        for request in element.tool_requests
    }
    emitted_requests: set[str] = set()
    emitted_responses: set[str] = set()
    fallbacks: list[ModelToolResponse] = []

    def flush_fallbacks() -> Generator[ConversationEvent]:
        # delivered responses follow the requests they answer, before any following content
        for response in fallbacks:
            if response.identifier not in emitted_responses:
                emitted_responses.add(response.identifier)
                yield ConversationEvent.tool_response(response)

        fallbacks.clear()

    for element in elements:
        if isinstance(element, ModelInput):
            yield from flush_fallbacks()
            for input_block in element.input:
                if isinstance(input_block, MultimodalContent):
                    yield input_block

                elif (
                    input_block.identifier in requested
                    and input_block.identifier not in emitted_responses
                ):
                    emitted_responses.add(input_block.identifier)
                    yield ConversationEvent.tool_response(input_block)

        else:
            assert isinstance(element, ModelOutput)  # nosec: B101
            for output_block in element.output:
                if isinstance(output_block, ModelToolRequest):
                    if output_block.identifier in emitted_requests:
                        continue  # reported both as a chunk and within a completed turn

                    if output_block.identifier in responded:
                        emitted_requests.add(output_block.identifier)
                        yield ConversationEvent.tool_request(output_block)

                    elif output_block.identifier in handled_responses:
                        # the provider did not report the response, use the delivered one
                        emitted_requests.add(output_block.identifier)
                        yield ConversationEvent.tool_request(output_block)
                        fallbacks.append(handled_responses[output_block.identifier])

                else:
                    yield from flush_fallbacks()
                    yield output_block

    yield from flush_fallbacks()


def _assistant_turn(
    elements: Sequence[ModelInput | ModelOutput],
    handled_responses: Mapping[str, ModelToolResponse],
) -> ConversationAssistantTurn | None:
    return ConversationAssistantTurn.of(*_assistant_turn_content(elements, handled_responses))
