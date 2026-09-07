import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

import pytest
from haiway import Meta

from draive import (
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    MultimodalContent,
    MultimodalContentPart,
    ProcessingEvent,
    TextContent,
    Toolbox,
    ToolsProvider,
    ctx,
    tool,
)


def _text_of(parts: Sequence[MultimodalContentPart]) -> str:
    return "".join(part.to_str() for part in parts)


def _multimodal_text_of(*elements: object) -> str:
    return MultimodalContent.of(*elements).to_str()


@pytest.mark.asyncio
async def test_empty_toolbox_model_tools_returns_model_tools_none() -> None:
    async with ctx.scope("test"):
        model_tools = Toolbox.empty.model_tools(iteration=0)

    assert model_tools == ModelTools.none


@pytest.mark.asyncio
async def test_handle_without_requests_yields_no_chunks() -> None:
    async with ctx.scope("test"):
        chunks = [chunk async for chunk in Toolbox.empty.handle(())]

    assert chunks == []


@pytest.mark.asyncio
async def test_handle_returns_error_response_for_unknown_tool() -> None:
    async with ctx.scope("test"):
        chunks = [
            chunk
            async for chunk in Toolbox.empty.handle(
                (ModelToolRequest.of("r1", tool="missing", arguments={}),)
            )
        ]

    assert len(chunks) == 1
    response = chunks[0]
    assert isinstance(response, ModelToolResponse)
    assert response.identifier == "r1"
    assert response.tool == "missing"
    assert response.status == "error"
    assert response.content.to_str() == "<error>Requested unknown tool `missing`</error>"


@pytest.mark.asyncio
async def test_handle_response_tool_streams_events_and_returns_accumulated_response() -> None:
    async with ctx.scope("test"):

        @tool
        async def lookup(value: str):
            yield ProcessingEvent.of("progress", f"checking:{value}")
            yield TextContent.of("A:")
            yield TextContent.of(value)

        chunks = [
            chunk
            async for chunk in Toolbox.of(lookup).handle(
                (ModelToolRequest.of("r1", tool="lookup", arguments={"value": "x"}),)
            )
        ]

    assert len(chunks) == 2
    event = chunks[0]
    response = chunks[1]
    assert isinstance(event, ProcessingEvent)
    assert event.event == "progress"
    assert event.content.to_str() == "checking:x"
    assert event.meta["tool"] == "lookup"
    assert event.meta["request"] == "r1"
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert response.content.to_str() == "A:x"


@pytest.mark.asyncio
async def test_handle_response_tool_returns_error_response_with_partial_result() -> None:
    async with ctx.scope("test"):

        @tool
        async def unstable():
            yield ProcessingEvent.of("progress", "started")
            yield TextContent.of("partial")
            raise RuntimeError("boom")

        chunks = [
            chunk
            async for chunk in Toolbox.of(unstable).handle(
                (ModelToolRequest.of("r1", tool="unstable", arguments={}),)
            )
        ]

    assert len(chunks) == 2
    assert isinstance(chunks[0], ProcessingEvent)
    response = chunks[1]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "error"
    assert (
        response.content.to_str() == "partial<error>Tool execution failed due to an error</error>"
    )


@pytest.mark.asyncio
async def test_handle_output_tool_yields_event_then_output_parts_then_response() -> None:
    async with ctx.scope("test"):

        @tool(handling="output")
        async def amplify(value: str):
            yield ProcessingEvent.of("progress", "starting")
            yield TextContent.of("OUT:")
            yield TextContent.of(value)

        chunks = [
            chunk
            async for chunk in Toolbox.of(amplify).handle(
                (ModelToolRequest.of("r1", tool="amplify", arguments={"value": "x"}),)
            )
        ]

    assert len(chunks) == 4
    event = chunks[0]
    output_parts = chunks[1:3]
    response = chunks[3]
    assert isinstance(event, ProcessingEvent)
    assert event.event == "progress"
    assert event.meta["tool"] == "amplify"
    assert event.meta["request"] == "r1"
    assert _multimodal_text_of(*output_parts) == "OUT:x"
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert response.content.to_str() == "OUT:x"


@pytest.mark.asyncio
async def test_handle_output_tool_returns_error_response_with_partial_result() -> None:
    async with ctx.scope("test"):

        @tool(handling="output")
        async def unstable_output():
            yield TextContent.of("OUT:")
            raise RuntimeError("boom")

        chunks = [
            chunk
            async for chunk in Toolbox.of(unstable_output).handle(
                (ModelToolRequest.of("r1", tool="unstable_output", arguments={}),)
            )
        ]

    assert len(chunks) == 2
    assert _multimodal_text_of(*chunks[:1]) == "OUT:"
    response = chunks[1]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "error"
    assert response.content.to_str() == "OUT:<error>Tool execution failed due to an error</error>"


@pytest.mark.asyncio
async def test_filtered_with_empty_tool_names_returns_empty_toolbox() -> None:
    async with ctx.scope("test"):

        @tool
        async def ping() -> str:
            return "pong"

        filtered = Toolbox.of(ping).filtered(tools=set())

    assert filtered.tools == {}


@pytest.mark.asyncio
async def test_model_tools_keeps_specific_suggestion_after_replacing_tool_instance() -> None:
    async with ctx.scope("test"):

        @tool
        async def ping() -> str:
            return "pong"

        toolbox = Toolbox.of(ping, suggesting=ping)
        updated = ping.updating(description="Updated description")

        model_tools = toolbox.with_tools(updated).model_tools(iteration=0)

    assert model_tools.selection == updated.specification


@pytest.mark.asyncio
async def test_tools_provider_toolbox_accepts_tool_suggestion() -> None:
    async with ctx.scope("test"):

        @tool
        async def ping() -> str:
            return "pong"

        async def load_tools() -> tuple[()]:
            return ()

        toolbox = await ToolsProvider(load_tools).toolbox(ping, suggesting=ping)

    assert toolbox.model_tools(iteration=0).selection == ping.specification


@pytest.mark.asyncio
async def test_tools_provider_toolbox_uses_provider_meta_by_default() -> None:
    async with ctx.scope("test"):

        async def load_tools() -> tuple[()]:
            return ()

        toolbox = await ToolsProvider(
            load_tools,
            meta=Meta.of({"provider": "value"}),
        ).toolbox()

    assert toolbox.meta == Meta.of({"provider": "value"})


@pytest.mark.asyncio
async def test_tools_provider_toolbox_allows_overriding_provider_meta() -> None:
    async with ctx.scope("test"):

        async def load_tools() -> tuple[()]:
            return ()

        toolbox = await ToolsProvider(
            load_tools,
            meta=Meta.of({"provider": "value"}),
        ).toolbox(meta={"call": "value"})

    assert toolbox.meta == Meta.of({"call": "value"})


@pytest.mark.asyncio
async def test_tool_updating_allows_clearing_meta() -> None:
    async with ctx.scope("test"):

        @tool(meta={"tag": "value"})
        async def ping() -> str:
            return "pong"

        updated = ping.updating(meta=Meta.empty)

    assert ping.meta == Meta.of({"tag": "value"})
    assert updated.meta == Meta.empty


@pytest.mark.asyncio
async def test_handle_abandoned_mid_stream_leaves_no_pending_callback_errors() -> None:
    # cancelling the tool tasks fires their done callback, which must not ask a
    # cancelled task for its exception - doing so raises into the event loop
    loop_errors: list[str] = []

    def record_loop_error(loop: object, context: Mapping[str, Any]) -> None:
        loop_errors.append(str(context.get("message")))

    @tool(name="slow")
    async def slow() -> str:
        await asyncio.sleep(5)
        return "done"

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(record_loop_error)
    try:
        async with ctx.scope("test"):
            stream = Toolbox.of(slow).handle(
                (
                    ModelToolRequest.of("r1", tool="slow", arguments={}),
                    ModelToolRequest.of("r2", tool="slow", arguments={}),
                )
            )
            pending = asyncio.ensure_future(anext(stream))
            await asyncio.sleep(0.01)  # let the tool tasks start
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending

            await stream.aclose()

        await asyncio.sleep(0)  # let any pending callback run

    finally:
        loop.set_exception_handler(previous_handler)

    assert loop_errors == []


@pytest.mark.asyncio
async def test_handle_waits_for_every_request_to_respond() -> None:
    # the merged stream must not end with the first finished tool - every request
    # has to deliver its response, including the ones still running at that time
    async with ctx.scope("test"):

        @tool(name="fast", handling="output")
        async def fast():
            yield TextContent.of("fast-out")

        @tool(name="slow", handling="output")
        async def slow():
            await asyncio.sleep(0.05)
            yield TextContent.of("slow-out")

        chunks = [
            chunk
            async for chunk in Toolbox.of(fast, slow).handle(
                (
                    ModelToolRequest.of("r1", tool="fast", arguments={}),
                    ModelToolRequest.of("r2", tool="slow", arguments={}),
                )
            )
        ]

    responses = [chunk for chunk in chunks if isinstance(chunk, ModelToolResponse)]
    assert [response.identifier for response in responses] == ["r1", "r2"]
    assert all(response.status == "success" for response in responses)

    outputs = [chunk for chunk in chunks if isinstance(chunk, MultimodalContentPart)]
    assert [chunk.meta["request"] for chunk in outputs] == ["r1", "r2"]
    assert _text_of(outputs) == "fast-outslow-out"


@pytest.mark.asyncio
async def test_handle_keeps_output_contiguous_per_request() -> None:
    # the first request producing output streams it directly, the output of the
    # others is accumulated and appended once the streaming one finishes
    async with ctx.scope("test"):
        first_started = asyncio.Event()
        second_started = asyncio.Event()

        @tool(name="first", handling="output")
        async def first():
            yield TextContent.of("f1")
            first_started.set()
            await second_started.wait()
            yield TextContent.of("f2")

        @tool(name="second", handling="output")
        async def second():
            await first_started.wait()
            yield TextContent.of("s1")
            second_started.set()
            yield TextContent.of("s2")

        chunks = [
            chunk
            async for chunk in Toolbox.of(first, second).handle(
                (
                    ModelToolRequest.of("r1", tool="first", arguments={}),
                    ModelToolRequest.of("r2", tool="second", arguments={}),
                )
            )
        ]

    outputs = [chunk for chunk in chunks if isinstance(chunk, MultimodalContentPart)]
    assert [chunk.to_str() for chunk in outputs] == ["f1", "f2", "s1", "s2"]
    assert [chunk.meta["request"] for chunk in outputs] == ["r1", "r1", "r2", "r2"]

    responses = [chunk for chunk in chunks if isinstance(chunk, ModelToolResponse)]
    assert {response.identifier for response in responses} == {"r1", "r2"}
    assert all(response.status == "success" for response in responses)
