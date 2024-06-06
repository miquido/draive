from asyncio import CancelledError, Task, sleep

from draive import ctx, with_timeout
from pytest import mark, raises


class FakeException(Exception):
    pass


@mark.asyncio
@ctx.wrap("test")
async def test_returns_result_when_returning_value():
    @with_timeout(3)
    async def long_running() -> int:
        return 42

    assert await long_running() == 42


@mark.asyncio
@ctx.wrap("test")
async def test_raises_with_error():
    @with_timeout(3)
    async def long_running() -> int:
        raise FakeException()

    with raises(FakeException):
        await long_running()


@mark.asyncio
@ctx.wrap("test")
async def test_raises_with_cancel():
    @with_timeout(3)
    async def long_running() -> int:
        await sleep(1)
        raise RuntimeError("Invalid state")

    task = Task(long_running())
    with raises(CancelledError):
        await sleep(0.01)
        task.cancel()
        await task


@mark.asyncio
@ctx.wrap("test")
async def test_raises_with_timeout():
    @with_timeout(0.01)
    async def long_running() -> int:
        await sleep(0.03)
        raise RuntimeError("Invalid state")

    with raises(TimeoutError):
        await long_running()
