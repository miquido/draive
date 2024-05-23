from asyncio import AbstractEventLoop, CancelledError, Future, get_running_loop
from collections import deque
from collections.abc import AsyncIterator
from typing import Self

__all__ = [
    "AsyncStream",
]


class AsyncStream[Element](AsyncIterator[Element]):
    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
    ) -> None:
        self._loop: AbstractEventLoop = loop or get_running_loop()
        self._buffer: deque[Element] = deque()
        self._waiting_queue: deque[Future[Element]] = deque()
        self._finish_exception: BaseException | None = None

    def __del__(self) -> None:
        while self._waiting_queue:
            waiting: Future[Element] = self._waiting_queue.popleft()
            if waiting.done():
                continue
            else:
                waiting.set_exception(CancelledError())

    @property
    def finished(self) -> bool:
        return self._finish_exception is not None

    def send(
        self,
        element: Element,
    ) -> None:
        if self.finished:
            raise RuntimeError("AsyncStream has been already finished")

        while self._waiting_queue:
            assert not self._buffer  # nosec: B101
            waiting: Future[Element] = self._waiting_queue.popleft()
            if waiting.done():
                continue
            else:
                waiting.set_result(element)
                break
        else:
            self._buffer.append(element)

    def finish(
        self,
        exception: BaseException | None = None,
    ) -> None:
        if self.finished:
            raise RuntimeError("AsyncStream has been already finished")
        self._finish_exception = exception or StopAsyncIteration()
        if self._buffer:
            assert self._waiting_queue is None  # nosec: B101
            return  # allow consuming buffer to the end
        while self._waiting_queue:
            waiting: Future[Element] = self._waiting_queue.popleft()
            if waiting.done():
                continue
            else:
                waiting.set_exception(self._finish_exception)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        if self._buffer:  # use buffer first
            return self._buffer.popleft()
        if finish_exception := self._finish_exception:  # check if finished
            raise finish_exception

        # create new waiting future
        future: Future[Element] = self._loop.create_future()
        self._waiting_queue.append(future)

        # wait for the result
        return await future
