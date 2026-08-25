from asyncio import sleep
from collections.abc import AsyncIterator

import pytest
from haiway import ctx

from draive import (
    GenerativeModel,
    ModelInput,
    ModelOutputChunk,
    ModelToolRequest,
    MultimodalContent,
    ProcessingEvent,
    Step,
    TextContent,
    Toolbox,
    ToolException,
    tool,
)


async def _endless_completion(released: list[str], **_: object) -> AsyncIterator[ModelOutputChunk]:
    async with ctx.scope("model.invocation"):
        try:
            index: int = 0
            while True:
                yield TextContent.of(f"chunk-{index}")
                index += 1
                await sleep(0)

        finally:
            released.append("released")


@pytest.mark.asyncio
async def test_step_stream_releases_model_completion_when_closed_early() -> None:
    released: list[str] = []

    def completion(**extra: object) -> AsyncIterator[ModelOutputChunk]:
        return _endless_completion(released, **extra)

    async with ctx.scope("test", GenerativeModel(generating=completion)):
        stream = Step.looping_completion(
            instructions="",
            output="text",
        ).stream((ModelInput.of(MultimodalContent.of("start")),))

        received: int = 0
        async for _ in stream:
            received += 1
            if received > 2:
                break  # stop before the completion ends

        assert received == 3
        assert released == []

        # closing the stream unwinds every layer down to the completion
        await stream.aclose()

    assert released == ["released"]


@pytest.mark.asyncio
async def test_step_stream_releases_model_completion_when_drained() -> None:
    released: list[str] = []

    async def completion(**_: object) -> AsyncIterator[ModelOutputChunk]:
        async with ctx.scope("model.invocation"):
            try:
                yield TextContent.of("done")

            finally:
                released.append("released")

    async with ctx.scope("test", GenerativeModel(generating=completion)):
        chunks = [
            chunk
            async for chunk in Step.looping_completion(
                instructions="",
                output="text",
            ).stream((ModelInput.of(MultimodalContent.of("start")),))
        ]

    assert [chunk.to_str() for chunk in chunks] == ["done"]
    assert released == ["released"]


@pytest.mark.asyncio
async def test_tools_handling_releases_pending_tool_when_closed_early() -> None:
    released: list[str] = []

    async with ctx.scope("test"):

        @tool(handling="output_stream")
        async def endless():
            try:
                index: int = 0
                while True:
                    yield f"chunk-{index}"
                    index += 1
                    await sleep(0)

            finally:
                released.append("released")

        stream = Toolbox.of(endless).handle(ModelToolRequest.of("r1", tool="endless", arguments={}))

        received: int = 0
        async for _ in stream:
            received += 1
            if received > 1:
                break  # stop before the tool ends

        assert released == []

        await stream.aclose()

    assert released == ["released"]


@pytest.mark.asyncio
async def test_tools_handling_releases_pending_response_tool_when_closed_early() -> None:
    released: list[str] = []

    async with ctx.scope("test"):

        @tool  # a response tool streams its events, the content comes as the final response
        async def endless():
            try:
                index: int = 0
                while True:
                    yield ProcessingEvent.of(f"event-{index}")
                    index += 1
                    await sleep(0)

            finally:
                released.append("released")

        stream = Toolbox.of(endless).handle(ModelToolRequest.of("r1", tool="endless", arguments={}))

        received: int = 0
        async for _ in stream:
            received += 1
            if received > 1:
                break  # stop before the tool ends

        assert released == []

        await stream.aclose()

    assert released == ["released"]


@pytest.mark.asyncio
async def test_toolbox_call_releases_tool_when_it_fails() -> None:
    released: list[str] = []

    async with ctx.scope("test"):

        @tool
        async def failing():
            try:
                yield "chunk"
                raise ValueError("tool failure")

            finally:
                released.append("released")

        with pytest.raises(ToolException):
            await Toolbox.of(failing).call("failing")

    assert released == ["released"]


@pytest.mark.asyncio
async def test_tool_rejects_non_generator_stream_function() -> None:
    async def stream() -> AsyncIterator[str]:
        yield "chunk"

    # returning an async iterable is not enough, only generators can be released
    def not_a_generator() -> AsyncIterator[str]:
        return stream()

    with pytest.raises(TypeError):
        tool(not_a_generator)  # pyright: ignore[reportArgumentType]
