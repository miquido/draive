from collections.abc import Generator

from pytest import mark, raises

from draive import MultimodalContent, cache, ctx, retry, tool


class FakeException(Exception):
    pass


@mark.asyncio
async def test_call_returns_result():
    async with ctx.scope("test"):
        executions: int = 0

        @tool
        async def compute(value: int) -> int:
            nonlocal executions
            executions += 1
            return value

        assert await compute(value=42) == 42
        assert executions == 1


@mark.asyncio
async def test_call_fails_on_error():
    async with ctx.scope("test"):
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
async def test_toolbox_call_returns_multimodal_content():
    async with ctx.scope("test"):
        executions: int = 0

        @tool
        async def compute(value: int) -> int:
            nonlocal executions
            executions += 1
            return value

        assert await compute.tool_call(
            "call_id",
            value=42,
        ) == MultimodalContent.of("42")
        assert executions == 1


@mark.asyncio
async def test_toolbox_call_returns_custom_content():
    async with ctx.scope("test"):
        executions: int = 0

        def custom_format(result: int) -> str:
            return f"Value:{result}"

        @tool(format_result=custom_format)
        async def compute(value: int) -> int:
            nonlocal executions
            executions += 1
            return value

        assert await compute.tool_call(
            "call_id",
            value=42,
        ) == MultimodalContent.of("Value:42")
        assert executions == 1


@mark.asyncio
async def test_retries_with_auto_retry():
    async with ctx.scope("test"):
        executions: int = 0

        @tool
        @retry
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
async def test_returns_cached_with_cache():
    async with ctx.scope("test"):

        def fake_random() -> Generator[int, None, None]:
            yield from range(0, 65536)

        @tool
        @cache
        async def compute(value: int) -> int:
            return next(fake_random())

        expected: int = await compute(value=42)
        assert await compute(value=42) == expected
