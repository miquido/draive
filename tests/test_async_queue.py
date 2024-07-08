from asyncio import CancelledError

import pytest
from draive import AsyncQueue, ctx
from pytest import raises


class FakeException(Exception):
    pass


@pytest.mark.asyncio
async def test_fails_when_stream_fails():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.enqueue(0)
    stream.finish(exception=FakeException())
    elements: int = 0
    with raises(FakeException):
        async for _ in stream:
            elements += 1

    assert elements == 1


@pytest.mark.asyncio
async def test_cancels_when_iteration_cancels():
    stream: AsyncQueue[int] = AsyncQueue()
    elements: int = 0
    with raises(CancelledError):
        ctx.cancel()
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
async def test_ends_when_stream_ends():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.finish()
    elements: int = 0
    async for _ in stream:
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
async def test_buffers_values_when_not_reading():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.enqueue(0)
    stream.enqueue(1)
    stream.enqueue(2)
    stream.enqueue(3)
    stream.finish()
    elements: int = 0

    async for _ in stream:
        elements += 1

    assert elements == 4


@pytest.mark.asyncio
async def test_delivers_buffer_when_streaming_fails():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.enqueue(0)
    stream.enqueue(1)
    stream.enqueue(2)
    stream.enqueue(3)
    stream.finish(exception=FakeException())
    elements: int = 0

    with raises(FakeException):
        async for _ in stream:
            elements += 1

    assert elements == 4


@pytest.mark.asyncio
async def test_delivers_updates_when_sending():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.enqueue(0)

    elements: list[int] = []

    async for element in stream:
        elements.append(element)
        if len(elements) < 10:
            stream.enqueue(element + 1)
        else:
            stream.finish()

    assert elements == list(range(0, 10))


@pytest.mark.asyncio
async def test_fails_when_sending_to_finished():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.finish()

    with raises(RuntimeError):
        stream.enqueue(42)


@pytest.mark.asyncio
async def test_ignores_when_finishing_when_finished():
    stream: AsyncQueue[int] = AsyncQueue()
    stream.finish()
    stream.finish()  # should not raise
