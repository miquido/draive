from asyncio import AbstractEventLoop, CancelledError, Event, Future, get_running_loop
from collections import deque
from collections.abc import AsyncIterator
from typing import Self

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
        self._pending: Future[Element] | None = None
        self._finished: Event = Event()

        freeze(self)

    def __del__(self) -> None:
        self.finish()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def enqueue(
        self,
        element: Element,
        /,
        *elements: Element,
    ) -> None:
        if self._finished.is_set():
            return  # ignore

        if pending := self._pending:
            assert not self._queue  # nosec: B101
            if not pending.done():
                pending.set_result(element)

            self._pending = None

        else:
            self._queue.append(element)

        self._queue.extend(elements)

    def cancel(self) -> None:
        if self._finished.is_set():
            return

        self._finished.set()

        while self._queue:  # clear the queue on cancel
            self._queue.popleft()

        if pending := self._pending:
            if not pending.done():
                pending.set_exception(CancelledError())

    def finish(self) -> None:
        if self._finished.is_set():
            return

        self._finished.set()

        if pending := self._pending:
            if not pending.done():
                pending.set_exception(StopAsyncIteration())

    async def wait(self) -> None:
        await self._finished.wait()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        if self._queue:  # check the queue, let it finish
            return self._queue.popleft()

        if self._finished.is_set():  # check if is finished
            raise StopAsyncIteration()

        # create a new future to wait for next
        assert self._pending is None, "Only a single queue iterator is supported!"  # nosec: B101
        future: Future[Element] = self._loop.create_future()
        self._pending = future

        # wait for the result
        return await future
