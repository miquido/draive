from collections.abc import Generator

from draive import MultimodalContent, auto_retry, cache, ctx, tool
from pytest import mark, raises


class FakeException(Exception):
    pass


@mark.asyncio
@ctx.wrap("test")
async def test_call_returns_result():
    executions: int = 0

    @tool
    async def compute(value: int) -> int:
        nonlocal executions
        executions += 1
        return value

    assert await compute(value=42) == 42
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_call_fails_on_error():
    executions: int = 0

    @tool
    async def compute(value: int) -> int:
        nonlocal executions
        executions += 1
        raise FakeException()

    with raises(FakeException):
        await compute(value=42)
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_toolbox_call_returns_multimodal_content():
    executions: int = 0

    @tool
    async def compute(value: int) -> int:
        nonlocal executions
        executions += 1
        return value

    assert await compute._toolbox_call(
        "call_id",
        arguments={
            "value": 42,
        },
    ) == MultimodalContent.of("42")
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_toolbox_call_returns_custom_content():
    executions: int = 0

    def custom_format(value: int) -> str:
        return f"Value:{value}"

    @tool(format_result=custom_format)
    async def compute(value: int) -> int:
        nonlocal executions
        executions += 1
        return value

    assert await compute._toolbox_call(
        "call_id",
        arguments={
            "value": 42,
        },
    ) == MultimodalContent.of("Value:42")
    assert executions == 1


@mark.asyncio
@ctx.wrap("test")
async def test_retries_with_auto_retry():
    executions: int = 0

    @tool
    @auto_retry
    async def compute(value: int) -> int:
        nonlocal executions
        executions += 1
        if executions == 1:
            raise FakeException()
        else:
            return value

    assert await compute(value=42) == 42
    assert executions == 2


@mark.asyncio
@ctx.wrap("test")
async def test_returns_cached_with_cache():
    def fake_random() -> Generator[int, None, None]:
        yield from range(0, 65536)

    @tool
    @cache
    async def compute(value: int) -> int:
        return next(fake_random())

    expected: int = await compute(value=42)
    assert await compute(value=42) == expected
