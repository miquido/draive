from asyncio import CancelledError, Task, sleep
from collections.abc import Callable, Generator
from time import sleep as sync_sleep

from draive import cache
from pytest import fixture, mark, raises


class FakeException(Exception):
    pass


@fixture
def fake_random() -> Callable[[], Generator[int, None, None]]:
    def random_next() -> Generator[int, None, None]:
        yield from range(0, 65536)

    return random_next


def test_returns_cached_value_with_same_argument(fake_random: Callable[[], int]):
    @cache
    def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = randomized("expected")
    assert randomized("expected") == expected


def test_returns_fresh_value_with_different_argument(fake_random: Callable[[], int]):
    @cache
    def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = randomized("expected")
    assert randomized("checked") != expected


def test_returns_fresh_value_with_limit_exceed(fake_random: Callable[[], int]):
    @cache(limit=1)
    def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = randomized("expected")
    randomized("different")
    assert randomized("expected") != expected


def test_returns_same_value_with_repeating_argument(fake_random: Callable[[], int]):
    @cache(limit=2)
    def randomized(_: str, /) -> int:
        return fake_random()

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


def test_returns_fresh_value_with_expiration_time_exceed(fake_random: Callable[[], int]):
    @cache(expiration=0.01)
    def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = randomized("expected")
    sync_sleep(0.02)
    assert randomized("expected") != expected


@mark.asyncio
async def test_async_returns_cached_value_with_same_argument(fake_random: Callable[[], int]):
    @cache
    async def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = await randomized("expected")
    assert await randomized("expected") == expected


@mark.asyncio
async def test_async_returns_fresh_value_with_different_argument(fake_random: Callable[[], int]):
    @cache
    async def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = await randomized("expected")
    assert await randomized("checked") != expected


@mark.asyncio
async def test_async_returns_fresh_value_with_limit_exceed(fake_random: Callable[[], int]):
    @cache(limit=1)
    async def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = await randomized("expected")
    await randomized("different")
    assert await randomized("expected") != expected


@mark.asyncio
async def test_async_returns_same_value_with_repeating_argument(fake_random: Callable[[], int]):
    @cache(limit=2)
    async def randomized(_: str, /) -> int:
        return fake_random()

    expected: int = await randomized("expected")
    await randomized("different")
    await randomized("expected")
    await randomized("more_different")
    await randomized("expected")
    await randomized("final_different")
    assert await randomized("expected") == expected


@mark.asyncio
async def test_async_returns_fresh_value_with_expiration_time_exceed(
    fake_random: Callable[[], int],
):
    @cache(expiration=0.01)
    async def randomized(_: str, /) -> int:
        return fake_random()

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
async def test_async_expiration_creates_new_task(fake_random: Callable[[], int]):
    @cache(expiration=0.01)
    async def randomized(_: str, /) -> int:
        await sleep(0.02)
        return fake_random()

    assert await randomized("expected") != await randomized("expected")


@mark.asyncio
async def test_async_fails_with_error():
    @cache(expiration=0.02)
    async def randomized(_: str, /) -> int:
        raise FakeException()

    with raises(FakeException):
        await randomized("expected")
