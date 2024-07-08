from asyncio import AbstractEventLoop, CancelledError, Future, get_running_loop
from collections import deque
from collections.abc import AsyncIterator
from typing import Self

__all__ = [
    "AsyncStream",
    "AsyncBufferedStream",
]


class AsyncStream[Element](AsyncIterator[Element]):
    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
    ) -> None:
        self._loop: AbstractEventLoop = loop or get_running_loop()
        self._ready: Future[None] = self._loop.create_future()
        self._ready.set_result(None)  # starting ready
        self._waiting: Future[Element] = self._loop.create_future()
        self._finish_exception: BaseException | None = None

    def __del__(self) -> None:
        if self.finished:
            return

        self._finish_exception = StopAsyncIteration()

        if not self._waiting.done():
            self._waiting.set_exception(self._finish_exception)

        if not self._ready.done():
            self._ready.set_exception(self._finish_exception)

    @property
    def finished(self) -> bool:
        return self._finish_exception is not None

    async def send(
        self,
        element: Element,
        /,
    ) -> None:
        while self._waiting.done() and not self.finished:
            try:
                # wait for readiness
                await self._ready

            finally:
                # create new waiting future afterwards
                self._ready = self._loop.create_future()

        if self._finish_exception is not None:
            raise self._finish_exception

        else:
            self._waiting.set_result(element)

    def finish(
        self,
        exception: BaseException | None = None,
    ) -> None:
        if self.finished:
            raise RuntimeError("AsyncStream has been already finished")

        self._finish_exception = exception or StopAsyncIteration()

        if not self._waiting.done():
            self._waiting.set_exception(self._finish_exception)

        if not self._ready.done():
            self._ready.set_exception(self._finish_exception)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        if self._finish_exception is not None:
            raise self._finish_exception

        try:
            # wait for the result
            return await self._waiting

        finally:
            # create new waiting future afterwards
            self._waiting = self._loop.create_future()
            # and notify readiness
            self._loop.call_soon(self._notify_ready)

    def _notify_ready(self) -> None:
        if self._ready.done():
            return

        self._ready.set_result(None)


class AsyncBufferedStream[Element](AsyncIterator[Element]):
    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
    ) -> None:
        self._loop: AbstractEventLoop = loop or get_running_loop()
        self._buffer: deque[Element] = deque()
        self._waiting: Future[Element] | None = None
        self._finish_exception: BaseException | None = None

    def __del__(self) -> None:
        self._finish_exception = CancelledError()

        if self._waiting and not self._waiting.done():
            self._waiting.set_exception(CancelledError())

    @property
    def finished(self) -> bool:
        return self._finish_exception is not None

    def send(
        self,
        element: Element,
        /,
    ) -> None:
        if self.finished:
            raise RuntimeError("AsyncStream has been already finished")

        if self._waiting and not self._waiting.done():
            self._waiting.set_result(element)

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
            return  # allow consuming buffer to the end

        if self._waiting and not self._waiting.done():
            self._waiting.set_exception(self._finish_exception)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        if self._buffer:  # use buffer first
            return self._buffer.popleft()

        if self._finish_exception is not None:
            raise self._finish_exception

        assert not self._waiting or self._waiting.done()  # nosec: B101
        # create new waiting future
        self._waiting = self._loop.create_future()

        # wait for the result
        return await self._waiting
