from asyncio import CancelledError, Task, sleep
from random import randint
from time import sleep as sync_sleep

from draive import cache
from pytest import mark, raises


class FakeException(Exception):
    pass


def test_returns_cached_value_with_same_argument():
    @cache
    def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = randomized("expected")
    assert randomized("expected") == expected


def test_returns_fresh_value_with_different_argument():
    @cache
    def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = randomized("expected")
    assert randomized("checked") != expected


def test_returns_fresh_value_with_limit_exceed():
    @cache(limit=1)
    def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = randomized("expected")
    randomized("different")
    assert randomized("expected") != expected


def test_returns_same_value_with_repeating_argument():
    @cache(limit=2)
    def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = randomized("expected")
    randomized("different")
    randomized("expected")
    randomized("more_different")
    randomized("expected")
    randomized("final_different")
    assert randomized("expected") == expected


def test_fails_with_error():
    @cache(expiration=0.02)
    def randomized(_: str, /) -> int:
        raise FakeException()

    with raises(FakeException):
        randomized("expected")


def test_returns_fresh_value_with_expiration_time_exceed():
    @cache(expiration=0.02)
    def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = randomized("expected")
    sync_sleep(0.02)
    assert randomized("expected") != expected


@mark.asyncio
async def test_async_returns_cached_value_with_same_argument():
    @cache
    async def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = await randomized("expected")
    assert await randomized("expected") == expected


@mark.asyncio
async def test_async_returns_fresh_value_with_different_argument():
    @cache
    async def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = await randomized("expected")
    assert await randomized("checked") != expected


@mark.asyncio
async def test_async_returns_fresh_value_with_limit_exceed():
    @cache(limit=1)
    async def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = await randomized("expected")
    await randomized("different")
    assert await randomized("expected") != expected


@mark.asyncio
async def test_async_returns_same_value_with_repeating_argument():
    @cache(limit=2)
    async def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = await randomized("expected")
    await randomized("different")
    await randomized("expected")
    await randomized("more_different")
    await randomized("expected")
    await randomized("final_different")
    assert await randomized("expected") == expected


@mark.asyncio
async def test_async_returns_fresh_value_with_expiration_time_exceed():
    @cache(expiration=0.02)
    async def randomized(_: str, /) -> int:
        return randint(-65536, 65535)

    expected: int = await randomized("expected")
    await sleep(0.02)
    assert await randomized("expected") != expected


@mark.asyncio
async def test_async_cancel_waiting_does_not_cancel_task():
    @cache
    async def randomized(_: str, /) -> int:
        try:
            await sleep(0.5)
            return 0
        except CancelledError:
            return 42

    expected: int = await randomized("expected")
    cancelled = Task(randomized("expected"))

    async def delayed_cancel() -> None:
        cancelled.cancel()

    Task(delayed_cancel())
    assert await randomized("expected") == expected


@mark.asyncio
async def test_async_expiration_does_not_cancel_task():
    @cache(expiration=0.01)
    async def randomized(_: str, /) -> int:
        try:
            await sleep(0.02)
            return 0
        except CancelledError:
            return 42

    assert await randomized("expected") == 0


@mark.asyncio
async def test_async_expiration_creates_new_task():
    @cache(expiration=0.01)
    async def randomized(_: str, /) -> int:
        await sleep(0.02)
        return randint(-65536, 65535)

    assert await randomized("expected") != await randomized("expected")


@mark.asyncio
async def test_async_fails_with_error():
    @cache(expiration=0.02)
    async def randomized(_: str, /) -> int:
        raise FakeException()

    with raises(FakeException):
        await randomized("expected")
