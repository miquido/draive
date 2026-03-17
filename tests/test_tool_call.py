from collections.abc import Generator

from pytest import mark, raises

from draive import MultimodalContent, cache, ctx, retry, tool


class FakeException(Exception):
    pass


async def _tool_call_content(tool_call, /, **arguments: object) -> MultimodalContent:
    return MultimodalContent.of(*[chunk async for chunk in tool_call(**arguments)])


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
        async def compute(value: int) -> str:
            nonlocal executions
            executions += 1
            return str(value)

        assert await _tool_call_content(compute.call, value=42) == MultimodalContent.of("42")
        assert executions == 1


@mark.asyncio
async def test_toolbox_call_returns_multimodal_parts():
    async with ctx.scope("test"):
        executions: int = 0

        @tool
        async def compute(value: int) -> int:
            nonlocal executions
            executions += 1
            return MultimodalContent.of("Value:", str(value))

        assert await _tool_call_content(compute.call, value=42) == MultimodalContent.of("Value:42")
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

        def fake_random() -> Generator[int]:
            yield from range(0, 65536)

        @tool
        @cache
        async def compute(value: int) -> int:
            return next(fake_random())

        expected: int = await compute(value=42)
        assert await compute(value=42) == expected


@mark.asyncio
async def test_toolbox_call_formats_mapping_as_json() -> None:
    async with ctx.scope("test"):

        @tool
        async def compute() -> str:
            return '{"a": 1, "b": 2}'

        assert await _tool_call_content(compute.call) == MultimodalContent.of('{"a": 1, "b": 2}')


@mark.asyncio
async def test_toolbox_call_returns_stringified_content() -> None:
    async with ctx.scope("test"):

        @tool
        async def compute() -> str:
            return '{"item": "value"}'

        assert await _tool_call_content(compute.call) == MultimodalContent.of('{"item": "value"}')
