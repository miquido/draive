from asyncio import CancelledError, Event, sleep
from collections.abc import Callable

import pytest
from draive import AsyncStream, AsyncStreamTask, ctx
from pytest import raises


class FakeException(Exception):
    pass


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_when_streaming_fails():
    async def stream_job(send_update: Callable[[int], None]):
        raise FakeException()

    elements: int = 0
    with raises(FakeException):
        async for _ in AsyncStreamTask(job=stream_job):
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_cancels_when_streaming_cancels():
    async def stream_job(send_update: Callable[[int], None]):
        await sleep(1)  # wait "forever", expecting cancellation

    elements: int = 0
    with raises(CancelledError):
        stream: AsyncStreamTask[int] = AsyncStreamTask(job=stream_job)
        ctx.cancel()
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_ends_when_iteration_ends():
    async def stream_job(send_update: Callable[[int], None]):
        await sleep(0)  # finish without updates

    elements: int = 0
    async for _ in AsyncStreamTask(job=stream_job):
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_buffers_values_when_not_reading():
    finish_event: Event = Event()

    async def stream_job(send_update: Callable[[int], None]):
        send_update(0)
        send_update(1)
        send_update(2)
        send_update(3)
        finish_event.set()

    stream: AsyncStreamTask[int] = AsyncStreamTask(job=stream_job)
    elements: int = 0
    await finish_event.wait()  # wait for all events to be sent
    async for _ in stream:
        elements += 1

    assert elements == 4


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_buffer_when_streaming_fails():
    finish_event: Event = Event()

    async def stream_job(send_update: Callable[[int], None]):
        send_update(0)
        send_update(1)
        send_update(2)
        send_update(3)
        try:
            raise FakeException()
        finally:
            finish_event.set()

    elements: int = 0
    stream: AsyncStreamTask[int] = AsyncStreamTask(job=stream_job)
    await finish_event.wait()  # wait for all events to be sent
    with raises(FakeException):
        async for _ in stream:
            elements += 1

    assert elements == 4


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_updates_when_sending():
    async def stream_job(send_update: Callable[[int], None]):
        for i in range(0, 10):
            send_update(i)

    elements: list[int] = []

    async for element in AsyncStreamTask(job=stream_job):
        elements.append(element)

    assert elements == list(range(0, 10))


@pytest.mark.asyncio
async def test_fails_when_sending_to_finished():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()

    with raises(RuntimeError):
        stream.send(42)


@pytest.mark.asyncio
async def test_fails_when_finishing_finished():
    stream: AsyncStream[int] = AsyncStream()
    stream.finish()

    with raises(RuntimeError):
        stream.finish()
