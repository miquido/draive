from asyncio import AbstractEventLoop, CancelledError, Future, get_running_loop
from collections import deque
from collections.abc import AsyncIterator
from typing import Never, Self

from draive.utils import freeze

__all__ = [
    "AsyncQueue",
]


class AsyncQueue[Element](AsyncIterator[Element]):
    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
    ) -> None:
        self._loop: AbstractEventLoop = loop or get_running_loop()
        self._queue: deque[Element] = deque()
        self._waiting: Future[Element] | None = None
        self._finished: Future[Never] = self._loop.create_future()

        freeze(self)

    def __del__(self) -> None:
        self.finish()

    @property
    def finished(self) -> bool:
        return self._finished.done()

    def enqueue(self, element: Element, /, *elements: Element) -> None:
        if self.finished:
            raise RuntimeError("AsyncQueue is already finished")

        if self._waiting is not None and not self._waiting.done():
            assert not self._queue  # nosec: B101
            self._waiting.set_result(element)

        else:
            self._queue.append(element)

        self._queue.extend(elements)

    def finish(
        self,
        exception: BaseException | None = None,
    ) -> None:
        if self.finished:
            return  # already finished, ignore

        finish_exception: BaseException = exception or StopAsyncIteration()

        self._finished.set_exception(finish_exception)

        if self._waiting is not None and not self._waiting.done():
            self._waiting.set_exception(finish_exception)

    def cancel(self) -> None:
        self.finish(exception=CancelledError())

    async def wait(self) -> None:
        try:
            await self._finished

        except Exception:  # nosec: B110
            pass  # ignore exceptions, only wait for completion

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        if self._queue:  # check the queue, let it finish
            return self._queue.popleft()

        if self.finished:  # check if is finished
            raise self._finished.exception()  # pyright: ignore[reportGeneralTypeIssues]

        # create a new future to wait for next
        assert self._waiting is None, "Only a single queue iterator is supported!"  # nosec: B101
        self._waiting = self._loop.create_future()

        try:
            # wait for the result
            return await self._waiting

        finally:
            # cleanup
            self._waiting = None
