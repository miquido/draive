from asyncio import CancelledError, Queue, QueueEmpty, Task, create_task
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from draive.openai.chat_tools import (
    OpenAIChatStreamingToolStatus,
    _execute_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
    _flush_chat_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from draive.openai.client import OpenAIClient
from draive.openai.config import OpenAIChatConfig
from draive.scope import ArgumentsTrace, ctx
from draive.tools import ToolException
from draive.types import Model, StreamingProgressUpdate, StringConvertible, Toolset

__all__ = [
    "OpenAIChatStream",
    "OpenAIChatStreamingMessagePart",
    "OpenAIChatStreamingPart",
    "_chat_stream",
]


class OpenAIChatStreamingMessagePart(Model):
    content: StringConvertible


OpenAIChatStreamingPart = OpenAIChatStreamingToolStatus | OpenAIChatStreamingMessagePart

OpenAIChatStream = AsyncIterator[OpenAIChatStreamingPart]


class _QueueStream(OpenAIChatStream):
    def __init__(
        self,
        task: Task[None],
        queue: Queue[OpenAIChatStreamingPart | BaseException | None],
    ):
        self._task: Task[None] = task
        self._task.add_done_callback(lambda task: queue.put_nowait(task.exception()))
        self._queue: Queue[OpenAIChatStreamingPart | BaseException | None] = queue

    async def __anext__(self) -> OpenAIChatStreamingPart:
        if self._task.done():
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
                    return cast(OpenAIChatStreamingPart, element)

        except CancelledError as exc:
            self._task.cancel()
            raise exc
        except QueueEmpty:
            raise StopAsyncIteration from None


async def _chat_stream(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    toolset: Toolset | None,
) -> OpenAIChatStream:
    progress_queue: Queue[OpenAIChatStreamingPart | BaseException | None] = Queue()
    return _QueueStream(
        task=create_task(
            _chat_streaming_completion(
                client=client,
                config=config,
                messages=messages,
                toolset=toolset,
                progress=lambda update: progress_queue.put_nowait(item=update),
            )
        ),
        queue=progress_queue,
    )


async def _chat_streaming_completion(
    *,
    client: OpenAIClient,
    config: OpenAIChatConfig,
    messages: list[ChatCompletionMessageParam],
    toolset: Toolset | None,
    progress: StreamingProgressUpdate[OpenAIChatStreamingPart],
) -> None:
    async with ctx.nested(
        "chat_stream",
        # TODO: add result trace
        ArgumentsTrace(messages=messages),
    ):
        completion_stream: AsyncStream[ChatCompletionChunk] = await client.chat_completion(
            config=config,
            messages=messages,
            tools=cast(
                list[ChatCompletionToolParam],
                toolset.available_tools if toolset else [],
            ),
            stream=True,
        )

        # load first chunk to decide what to do next
        head: ChatCompletionChunk = await anext(completion_stream)

        if not head.choices:
            raise ToolException("Invalid OpenAI completion - missing deltas!", head)

        completion_head: ChoiceDelta = head.choices[0].delta

        # TODO: record token usage - openAI does not provide usage insight when streaming
        # (or makes it differently than when using regular response and couldn't find it)

        if completion_head.tool_calls is not None and (toolset := toolset):
            tool_calls: list[ChatCompletionMessageToolCall] = await _flush_chat_tool_calls(
                tool_calls=completion_head.tool_calls,
                completion_stream=completion_stream,
            )
            messages.extend(
                await _execute_chat_tool_calls(
                    tool_calls=tool_calls,
                    toolset=toolset,
                    progress=lambda update: progress(update),
                )
            )

        elif completion_head.content is not None:
            return await _stream_completion(
                head=completion_head.content,
                tail=completion_stream,
                progress=progress,
            )

        else:
            raise ToolException("Invalid OpenAI completion", completion_head)

    # recursion outside of context
    await _chat_streaming_completion(
        client=client,
        config=config,
        messages=messages,
        toolset=toolset,
        progress=progress,
    )


async def _stream_completion(
    head: str,
    tail: AsyncStream[ChatCompletionChunk],
    progress: StreamingProgressUpdate[OpenAIChatStreamingPart],
) -> None:
    if head:  # provide head / first part if not empty
        progress(update=OpenAIChatStreamingMessagePart(content=head))

    async for part in tail:
        # we are always requesting single result - no need to take care of indices
        part_text: str = part.choices[0].delta.content or ""
        if not part_text:
            continue  # skip empty parts
        progress(update=OpenAIChatStreamingMessagePart(content=part_text))
