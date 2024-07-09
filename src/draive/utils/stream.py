from asyncio import AbstractEventLoop, CancelledError, Future, get_running_loop
from collections.abc import AsyncIterable
from typing import Never, Self

from draive.utils.freeze import freeze

__all__ = [
    "AsyncStream",
]


class AsyncStream[Element](AsyncIterable[Element]):
    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
    ) -> None:
        self._loop: AbstractEventLoop = loop or get_running_loop()
        self._ready: Future[None] = self._loop.create_future()
        self._waiting: Future[Element] | None = None
        self._finished: Future[Never] = self._loop.create_future()

        freeze(self)

    def __del__(self) -> None:
        self.finish()

    @property
    def finished(self) -> bool:
        return self._finished.done()

    async def send(
        self,
        element: Element,
        /,
    ) -> None:
        while self._waiting is None or self._waiting.done():
            if self.finished:
                raise self._finished.exception()  # pyright: ignore[reportGeneralTypeIssues]

            try:
                # wait for readiness
                await self._ready

            finally:
                # create new waiting future afterwards
                self._ready = self._loop.create_future()

        if self.finished:
            raise self._finished.exception()  # pyright: ignore[reportGeneralTypeIssues]

        else:
            self._waiting.set_result(element)

    def finish(
        self,
        exception: BaseException | None = None,
    ) -> None:
        if self.finished:
            return  # already finished, ignore

        finish_exception: BaseException = exception or StopAsyncIteration()

        if not self._ready.done():
            self._ready.set_result(None)

        if self._waiting is not None and not self._waiting.done():
            self._waiting.set_exception(finish_exception)

        self._finished.set_exception(finish_exception)

    def cancel(self) -> None:
        self.finish(exception=CancelledError())

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        assert self._waiting is None, "AsyncStream can't be reused"  # nosec: B101

        if self.finished:
            raise self._finished.exception()  # pyright: ignore[reportGeneralTypeIssues]

        try:
            assert not self._ready.done()  # nosec: B101
            # create new waiting future
            self._waiting = self._loop.create_future()
            # and notify readiness
            self._ready.set_result(None)
            # and wait for the result
            return await self._waiting

        finally:
            # cleanup waiting future
            self._waiting = None
            if not self._ready.done():  # ensure not hanging on untracked future
                self._ready.set_result(None)
            # and create new waiting future afterwards
            self._ready = self._loop.create_future()
