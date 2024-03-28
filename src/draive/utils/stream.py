from asyncio import (
    AbstractEventLoop,
    CancelledError,
    Event,
    Future,
    Task,
    get_running_loop,
)
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Generic, Self, TypeVar

from draive.scope import ctx
from draive.types import Model, UpdateSend

__all__ = [
    "AsyncStream",
    "AsyncStreamTask",
]

_Element = TypeVar(
    "_Element",
    bound=Model | str,
)


class AsyncStream(Generic[_Element], AsyncIterator[_Element]):
    def __init__(self) -> None:
        self._buffer: deque[_Element] = deque()
        self._waiting: Future[_Element] | None = None
        self._finished: Event = Event()
        self._loop: AbstractEventLoop | None = None

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def send(
        self,
        update: _Element,
    ) -> None:
        if self._finished.is_set():
            raise ValueError("AsyncStream has been already finished")
        if waiting := self._waiting:
            assert not self._buffer  # nosec: B101
            waiting.set_result(update)
            self._waiting = None
        else:
            self._buffer.append(update)

    def finish(
        self,
        exception: BaseException | None = None,
    ) -> None:
        if self._finished.is_set():
            raise ValueError("AsyncStream has been already finished")
        self._finished.set()
        if self._buffer:
            if exception is None:
                return  # allow consuming buffer to the end
            self._buffer.clear()  # when failed propagate the issue
        if waiting := self._waiting:
            waiting.set_exception(exception or StopAsyncIteration)
            self._waiting = None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> _Element:
        if self._waiting is not None:
            raise RuntimeError("AsyncStream reuse is forbidden")
        if self._buffer:
            return self._buffer.popleft()
        if self._finished.is_set():
            raise StopAsyncIteration

        self._waiting = get_running_loop().create_future()

        return await self._waiting


class AsyncStreamTask(Generic[_Element], AsyncIterator[_Element]):
    def __init__(
        self,
        job: Callable[[UpdateSend[_Element]], Coroutine[Any, Any, None]],
    ) -> None:
        stream: AsyncStream[_Element] = AsyncStream()
        self._stream: AsyncStream[_Element] = stream
        self._task: Task[None] = ctx.spawn_task(job, stream.send)
        self._task.add_done_callback(lambda task: stream.finish(task.exception()))

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> _Element:
        try:
            return await self._stream.__anext__()
        except CancelledError as exc:
            self._task.cancel()
            raise exc
        except StopAsyncIteration as exc:
            if error := self._task.exception():
                raise error from None
            else:
                raise exc
