from asyncio import CancelledError
from collections.abc import AsyncIterable, Generator
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Final

import pytest
from draive import ctx
from pytest import raises

EXECUTOR: Final[Executor | None] = ThreadPoolExecutor(max_workers=1)


class FakeException(Exception):
    pass


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_fails_when_streaming_fails():
    def stream_job() -> Generator[int]:
        raise FakeException()

    elements: int = 0
    with raises(FakeException):
        async for _ in ctx.stream_sync(stream_job(), executor=EXECUTOR):
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_cancels_when_streaming_cancels():
    def stream_job() -> Generator[int]:
        while False:
            yield  # finish without updates
        raise CancelledError()

    elements: int = 0
    with raises(CancelledError):
        stream: AsyncIterable[int] = ctx.stream_sync(stream_job(), executor=EXECUTOR)
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_cancels_when_task_cancels():
    def stream_job() -> Generator[int]:
        while False:
            yield  # finish without updates

    elements: int = 0
    with raises(CancelledError):
        stream: AsyncIterable[int] = ctx.stream_sync(stream_job(), executor=EXECUTOR)
        ctx.cancel()
        async for _ in stream:
            elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_ends_when_iteration_ends():
    def stream_job() -> Generator[int]:
        while False:
            yield  # finish without updates

    elements: int = 0
    async for _ in ctx.stream_sync(stream_job(), executor=EXECUTOR):
        elements += 1

    assert elements == 0


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_values_from_generator():
    def stream_job() -> Generator[int]:
        yield 0
        yield 1
        yield 2
        yield 3

    stream: AsyncIterable[int] = ctx.stream_sync(stream_job(), executor=EXECUTOR)
    elements: list[int] = []

    async for element in stream:
        elements.append(element)

    assert elements == [0, 1, 2, 3]


@pytest.mark.asyncio
@ctx.wrap("test")
async def test_delivers_errors_from_generator():
    def stream_job() -> Generator[int]:
        yield 0
        yield 1
        yield 2

        raise FakeException()

        yield 3

    stream: AsyncIterable[int] = ctx.stream_sync(stream_job(), executor=EXECUTOR)
    elements: list[int] = []

    with raises(FakeException):
        async for element in stream:
            elements.append(element)

    assert elements == [0, 1, 2]
