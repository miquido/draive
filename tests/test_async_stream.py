from asyncio import CancelledError

import pytest
from draive import AsyncStream, ctx
from pytest import raises


class FakeException(Exception):
    pass


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_when_stream_fails():
    stream: AsyncStream[int] = AsyncStream()
    ctx.spawn_subtask(stream.send, 0)
    elements: int = 0
    with raises(FakeException):
        async for _ in stream:
            elements += 1
            stream.finish(exception=FakeException())

    assert elements == 1


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_cancels_when_iteration_cancels():
    stream: AsyncStream[int] = AsyncStream()
    elements: int = 0
    with raises(CancelledError):
        ctx.cancel()
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_ends_when_stream_ends():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()
    elements: int = 0
    async for _ in stream:
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_finishes_without_buffer():
    stream: AsyncStream[int] = AsyncStream()
    ctx.spawn_subtask(stream.send, 0)
    ctx.spawn_subtask(stream.send, 1)
    ctx.spawn_subtask(stream.send, 2)
    ctx.spawn_subtask(stream.send, 3)
    stream.finish()
    elements: int = 0

    async for _ in stream:
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_without_buffer():
    stream: AsyncStream[int] = AsyncStream()
    ctx.spawn_subtask(stream.send, 0)
    ctx.spawn_subtask(stream.send, 1)
    ctx.spawn_subtask(stream.send, 2)
    ctx.spawn_subtask(stream.send, 3)
    stream.finish(exception=FakeException())
    elements: int = 0

    with raises(FakeException):
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_updates_when_sending():
    stream: AsyncStream[int] = AsyncStream()
    ctx.spawn_subtask(stream.send, 0)

    elements: list[int] = []

    async for element in stream:
        elements.append(element)
        if len(elements) < 10:
            ctx.spawn_subtask(stream.send, element + 1)
        else:
            stream.finish()

    assert elements == list(range(0, 10))


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_when_sending_to_finished():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()

    with raises(StopAsyncIteration):
        await stream.send(42)


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_when_sending_to_failed():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish(exception=FakeException())

    with raises(FakeException):
        await stream.send(42)


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_ignores_when_finishing_when_finished():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()
    stream.finish()  # should not raise
