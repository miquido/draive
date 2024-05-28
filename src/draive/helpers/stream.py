from asyncio import Task
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Self

from draive.scope import ctx
from draive.utils import AsyncStream

__all__ = [
    "AsyncStreamTask",
]


class AsyncStreamTask[Element](AsyncIterator[Element]):
    def __init__(
        self,
        job: Callable[[Callable[[Element], None]], Coroutine[None, None, None]],
    ) -> None:
        stream: AsyncStream[Element] = AsyncStream()
        self._stream: AsyncStream[Element] = stream

        async def streaming() -> None:
            try:
                await job(stream.send)
            except Exception as exc:
                stream.finish(exc)
            else:
                stream.finish()

        self._task: Task[None] = ctx.spawn_task(streaming)

    def __del__(self) -> None:
        self._task.cancel()

    def cancel(self) -> None:
        self._task.cancel()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Element:
        return await self._stream.__anext__()
