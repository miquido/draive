from asyncio import CancelledError, Task, sleep
from time import time
from unittest import TestCase

from draive import MetricsTrace, auto_retry, ctx
from pytest import mark, raises


class FakeException(Exception):
    pass


@mark.asyncio
@ctx.wrap("test")
async def test_returns_value_without_errors():
    executions: int = 0

    @auto_retry
    def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        return value

    assert compute("expected") == "expected"
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_retries_with_errors():
    executions: int = 0

    @auto_retry
    def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        if executions == 1:
            raise FakeException()
        else:
            return value

    assert compute("expected") == "expected"
    assert executions == 2


@mark.asyncio
@ctx.wrap("test")
async def test_logs_issue_with_errors():
    executions: int = 0
    test_case = TestCase()

    @auto_retry
    def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        if executions == 1:
            raise FakeException("fake")
        else:
            return value

    metrics_trace: MetricsTrace = ctx._current_metrics()
    with test_case.assertLogs() as logs:
        compute("expected")
        assert executions == 2
        assert logs.output == [
            f"ERROR:test:[{metrics_trace}] Attempting to retry {compute.__name__}"
            f" which failed due to an error: {FakeException("fake")}"
        ]


@mark.asyncio
@ctx.wrap("test")
async def test_fails_with_exceeding_errors():
    executions: int = 0

    @auto_retry(limit=1)
    def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        raise FakeException()

    with raises(FakeException):
        compute("expected")
    assert executions == 2


@mark.asyncio
@ctx.wrap("test")
async def test_fails_with_cancellation():
    executions: int = 0

    @auto_retry(limit=1)
    def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        raise CancelledError()

    with raises(CancelledError):
        compute("expected")
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_async_returns_value_without_errors():
    executions: int = 0

    @auto_retry
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        return value

    assert await compute("expected") == "expected"
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_async_retries_with_errors():
    executions: int = 0

    @auto_retry
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        if executions == 1:
            raise FakeException()
        else:
            return value

    assert await compute("expected") == "expected"
    assert executions == 2


@mark.asyncio
@ctx.wrap("test")
async def test_async_fails_with_exceeding_errors():
    executions: int = 0

    @auto_retry(limit=1)
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        raise FakeException()

    with raises(FakeException):
        await compute("expected")
    assert executions == 2


@mark.asyncio
@ctx.wrap("test")
async def test_async_fails_with_cancellation():
    executions: int = 0

    @auto_retry(limit=1)
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        raise CancelledError()

    with raises(CancelledError):
        await compute("expected")
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_async_fails_when_cancelled():
    executions: int = 0

    @auto_retry(limit=1)
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        await sleep(1)
        return value

    with raises(CancelledError):
        task = Task(compute("expected"))
        await sleep(0.02)
        task.cancel()
        await task
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_async_uses_delay_with_errors():
    executions: int = 0

    @auto_retry(limit=2, delay=0.05)
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        raise FakeException()

    time_start: float = time()
    with raises(FakeException):
        await compute("expected")
    assert (time() - time_start) >= 0.1
    assert executions == 3


@mark.asyncio
@ctx.wrap("test")
async def test_async_uses_computed_delay_with_errors():
    executions: int = 0

    @auto_retry(limit=2, delay=lambda attempt: attempt * 0.035)
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        raise FakeException()

    time_start: float = time()
    with raises(FakeException):
        await compute("expected")
    assert (time() - time_start) >= 0.1
    assert executions == 3


@mark.asyncio
@ctx.wrap("test")
async def test_async_logs_issue_with_errors():
    executions: int = 0
    test_case = TestCase()

    @auto_retry
    async def compute(value: str, /) -> str:
        nonlocal executions
        executions += 1
        if executions == 1:
            raise FakeException("fake")
        else:
            return value

    metrics_trace: MetricsTrace = ctx._current_metrics()
    with test_case.assertLogs() as logs:
        await compute("expected")
        assert executions == 2
        assert logs.output[0].startswith(
            f"ERROR:test:[{metrics_trace}] Attempting to retry {compute.__name__}"
            " which failed due to an error"
        )
