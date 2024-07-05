from asyncio import CancelledError, sleep
from collections.abc import AsyncGenerator, AsyncIterator

import pytest
from draive import AsyncStream, ctx
from pytest import raises


class FakeException(Exception):
    pass


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_when_streaming_fails():
    def stream_job() -> AsyncGenerator[int, None]:
        raise FakeException()

    elements: int = 0
    with raises(FakeException):
        async for _ in ctx.stream(stream_job()):
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_cancels_when_streaming_cancels():
    async def stream_job() -> AsyncGenerator[int, None]:
        while False:
            yield  # finish without updates
        raise CancelledError()

    elements: int = 0
    with raises(CancelledError):
        stream: AsyncIterator[int] = ctx.stream(stream_job())
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_cancels_when_task_cancels():
    async def stream_job() -> AsyncGenerator[int, None]:
        yield 0
        await sleep(1)  # wait "forever", expecting cancellation

    elements: int = 0
    with raises(CancelledError):
        stream: AsyncIterator[int] = ctx.stream(stream_job())
        ctx.cancel()
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_ends_when_iteration_ends():
    async def stream_job() -> AsyncGenerator[int, None]:
        while False:
            yield  # finish without updates

    elements: int = 0
    async for _ in ctx.stream(stream_job()):
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_values_from_generator():
    async def stream_job() -> AsyncGenerator[int, None]:
        yield 0
        yield 1
        yield 2
        yield 3

    stream: AsyncIterator[int] = ctx.stream(stream_job())
    elements: list[int] = []

    async for element in stream:
        elements.append(element)

    assert elements == [0, 1, 2, 3]


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_errors_from_generator():
    async def stream_job() -> AsyncGenerator[int, None]:
        yield 0
        yield 1
        yield 2

        raise FakeException()

        yield 3

    stream: AsyncIterator[int] = ctx.stream(stream_job())
    elements: list[int] = []

    with raises(FakeException):
        async for element in stream:
            elements.append(element)

    assert elements == [0, 1, 2]


@pytest.mark.asyncio
async def test_fails_when_sending_to_finished():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()

    with raises(StopAsyncIteration):
        await stream.send(42)


@pytest.mark.asyncio
async def test_fails_when_finishing_finished():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()

    with raises(RuntimeError):
        stream.finish()
