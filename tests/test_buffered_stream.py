from asyncio import CancelledError

import pytest
from draive import AsyncBufferedStream, ctx
from pytest import raises


class FakeException(Exception):
    pass


@pytest.mark.asyncio
async def test_fails_when_stream_fails():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.send(0)
    stream.finish(exception=FakeException())
    elements: int = 0
    with raises(FakeException):
        async for _ in stream:
            elements += 1

    assert elements == 1


@pytest.mark.asyncio
async def test_cancels_when_iteration_cancels():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    elements: int = 0
    with raises(CancelledError):
        ctx.cancel()
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
async def test_ends_when_stream_ends():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.finish()
    elements: int = 0
    async for _ in stream:
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
async def test_buffers_values_when_not_reading():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.send(0)
    stream.send(1)
    stream.send(2)
    stream.send(3)
    stream.finish()
    elements: int = 0

    async for _ in stream:
        elements += 1

    assert elements == 4


@pytest.mark.asyncio
async def test_delivers_buffer_when_streaming_fails():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.send(0)
    stream.send(1)
    stream.send(2)
    stream.send(3)
    stream.finish(exception=FakeException())
    elements: int = 0

    with raises(FakeException):
        async for _ in stream:
            elements += 1

    assert elements == 4


@pytest.mark.asyncio
async def test_delivers_updates_when_sending():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.send(0)

    elements: list[int] = []

    async for element in stream:
        elements.append(element)
        if len(elements) < 10:
            stream.send(element + 1)
        else:
            stream.finish()

    assert elements == list(range(0, 10))


@pytest.mark.asyncio
async def test_fails_when_sending_to_finished():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.finish()

    with raises(RuntimeError):
        stream.send(42)


@pytest.mark.asyncio
async def test_fails_when_finishing_finished():
    stream: AsyncBufferedStream[int] = AsyncBufferedStream()
    stream.finish()

    with raises(RuntimeError):
        stream.finish()
