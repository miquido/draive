from asyncio import CancelledError, Queue, QueueEmpty, Task, create_task
from datetime import datetime
from typing import Literal, cast, overload

from draive.openai.chat import openai_chat_completion
from draive.openai.chat_stream import OpenAIChatStreamingMessagePart
from draive.openai.chat_tools import OpenAIChatStreamingToolStatus
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.types import (
    ConversationMessage,
    ConversationResponseStream,
    ConversationStreamingAction,
    ConversationStreamingActionStatus,
    ConversationStreamingPart,
    Memory,
    StreamingProgressUpdate,
    StringConvertible,
    Toolset,
)

__all__ = [
    "openai_conversation_completion",
]


@overload
async def openai_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: Literal[True],
) -> ConversationResponseStream:
    ...


@overload
async def openai_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
) -> ConversationMessage:
    ...


async def openai_conversation_completion(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    stream: bool = False,
) -> ConversationResponseStream | ConversationMessage:
    if stream:  # pyright does not get the type right here, have to use literals
        progress_queue: Queue[ConversationStreamingPart | BaseException | None] = Queue()
        return _ResponseStream(
            task=create_task(
                _openai_conversation_stream(
                    instruction=instruction,
                    input=input,
                    memory=memory,
                    toolset=toolset,
                    progress=lambda update: progress_queue.put_nowait(item=update),
                )
            ),
            queue=progress_queue,
        )

    else:
        async with ctx.nested(
            "openai_conversation_stream",
            ArgumentsTrace(message=input),
        ):
            user_message: ConversationMessage
            if isinstance(input, ConversationMessage):
                user_message = input

            else:
                user_message = ConversationMessage(
                    timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    author="user",
                    content=input,
                )

            history: list[ConversationMessage]

            if memory:
                history = await memory.recall()
            else:
                history = []

            response: ConversationMessage = ConversationMessage(
                timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                author="assistant",
                content=await openai_chat_completion(
                    instruction=instruction,
                    input=user_message.content,
                    history=history,
                    toolset=toolset,
                ),
            )

            if memory:
                await memory.remember(
                    [
                        *history,
                        user_message,
                        response,
                    ],
                )
            return response


class _ResponseStream(ConversationResponseStream):
    def __init__(
        self,
        task: Task[None],
        queue: Queue[ConversationStreamingPart | BaseException | None],
    ):
        self._task: Task[None] = task
        self._task.add_done_callback(lambda task: queue.put_nowait(task.exception()))
        self._queue: Queue[ConversationStreamingPart | BaseException | None] = queue

    async def __anext__(self) -> ConversationStreamingPart:
        if self._task.done() and self._queue.empty():
            if error := self._task.exception():
                raise error
            else:
                raise StopAsyncIteration
        try:
            match await self._queue.get():
                case None:
                    self._queue.task_done()
                    raise StopAsyncIteration

                case exception if isinstance(exception, BaseException):
                    self._queue.task_done()
                    raise exception

                case element:
                    self._queue.task_done()
                    return cast(ConversationStreamingPart, element)

        except CancelledError as exc:
            self._task.cancel()
            raise exc
        except QueueEmpty:
            raise StopAsyncIteration from None


async def _openai_conversation_stream(
    *,
    instruction: str,
    input: ConversationMessage | StringConvertible,  # noqa: A002
    memory: Memory[ConversationMessage] | None = None,
    toolset: Toolset | None = None,
    progress: StreamingProgressUpdate[ConversationStreamingPart],
) -> None:
    async with ctx.nested(
        "openai_conversation_stream",
        ArgumentsTrace(message=input),
    ):
        user_message: ConversationMessage
        if isinstance(input, ConversationMessage):
            user_message = input

        else:
            user_message = ConversationMessage(
                timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                author="user",
                content=input,
            )

        history: list[ConversationMessage]

        if memory:
            history = await memory.recall()
        else:
            history = []

        actions: dict[str, ConversationStreamingAction] = {}
        response: ConversationMessage = ConversationMessage(
            author="assistant",
            content="",
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        async for part in await openai_chat_completion(
            instruction=instruction,
            input=user_message.content,
            history=history,
            toolset=toolset,
            stream=True,
        ):
            match part:
                case OpenAIChatStreamingMessagePart(content=content):
                    response = response.updated(content=response.content_str + content.__str__())

                case OpenAIChatStreamingToolStatus(id=tool_id, name=tool_name, status=status):
                    actions[tool_id] = ConversationStreamingAction(
                        id=tool_id,
                        action="TOOL_CALL",
                        name=tool_name,
                        status=ConversationStreamingActionStatus(
                            current=status,
                        ),
                    )
            progress(
                ConversationStreamingPart(
                    actions=list(actions.values()),
                    message=response,
                )
            )

        await ctx.record(ResultTrace(response))

        if memory:
            await memory.remember(
                [
                    *history,
                    user_message,
                    response,
                ],
            )
