import asyncio
from collections.abc import Sequence

import pytest
from haiway import Meta

from draive import (
    ModelToolDetachedHandling,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
    MultimodalContent,
    MultimodalContentPart,
    ProcessingEvent,
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
        model_tools = Toolbox.empty.model_tools()

    assert model_tools == ModelTools.none


@pytest.mark.asyncio
async def test_handle_without_requests_yields_no_chunks() -> None:
    async with ctx.scope("test"):
        chunks = [chunk async for chunk in Toolbox.empty.handle()]

    assert chunks == []


@pytest.mark.asyncio
async def test_handle_returns_error_response_for_unknown_tool() -> None:
    async with ctx.scope("test"):
        chunks = [
            chunk
            async for chunk in Toolbox.empty.handle(
                ModelToolRequest.of("r1", tool="missing", arguments={})
            )
        ]

    assert len(chunks) == 1
    response = chunks[0]
    assert isinstance(response, ModelToolResponse)
    assert response.identifier == "r1"
    assert response.tool == "missing"
    assert response.handling == "response"
    assert response.status == "error"
    assert response.result.to_str() == "ERROR: Unknown tool"


@pytest.mark.asyncio
async def test_handle_response_tool_streams_events_and_returns_accumulated_response() -> None:
    async with ctx.scope("test"):

        @tool
        async def lookup(value: str):
            yield ProcessingEvent.of("progress", f"checking:{value}")
            yield "A:"
            yield value

        chunks = [
            chunk
            async for chunk in Toolbox.of(lookup).handle(
                ModelToolRequest.of("r1", tool="lookup", arguments={"value": "x"})
            )
        ]

    assert len(chunks) == 2
    event = chunks[0]
    response = chunks[1]
    assert isinstance(event, ProcessingEvent)
    assert event.event == "progress"
    assert event.content.to_str() == "checking:x"
    assert event.meta["tool"] == "lookup"
    assert event.meta["identifier"] == "r1"
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert response.handling == "response"
    assert response.result.to_str() == "A:x"


@pytest.mark.asyncio
async def test_handle_response_tool_returns_error_response_with_partial_result() -> None:
    async with ctx.scope("test"):

        @tool
        async def unstable():
            yield ProcessingEvent.of("progress", "started")
            yield "partial"
            raise RuntimeError("boom")

        chunks = [
            chunk
            async for chunk in Toolbox.of(unstable).handle(
                ModelToolRequest.of("r1", tool="unstable", arguments={})
            )
        ]

    assert len(chunks) == 2
    assert isinstance(chunks[0], ProcessingEvent)
    response = chunks[1]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "error"
    assert response.handling == "response"
    assert response.result.to_str() == "partial"


@pytest.mark.asyncio
async def test_handle_detached_tool_returns_detach_message_and_executes_in_background() -> None:
    executed = asyncio.Event()

    async with ctx.scope("test"):

        @tool(handling=ModelToolDetachedHandling(detach_message="working"))
        async def background(value: str) -> str:
            await asyncio.sleep(0)
            executed.set()
            return value.upper()

        chunks = [
            chunk
            async for chunk in Toolbox.of(background).handle(
                ModelToolRequest.of("r1", tool="background", arguments={"value": "x"})
            )
        ]
        await asyncio.wait_for(executed.wait(), timeout=1)

    assert len(chunks) == 1
    response = chunks[0]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert isinstance(response.handling, ModelToolDetachedHandling)
    assert response.result.to_str() == "working"


@pytest.mark.asyncio
async def test_handle_detached_tool_keeps_success_response_when_background_fails() -> None:
    attempted = asyncio.Event()

    async with ctx.scope("test"):

        @tool(handling=ModelToolDetachedHandling(detach_message="queued"))
        async def failing_background() -> str:
            attempted.set()
            raise RuntimeError("boom")

        chunks = [
            chunk
            async for chunk in Toolbox.of(failing_background).handle(
                ModelToolRequest.of("r1", tool="failing_background", arguments={})
            )
        ]
        await asyncio.wait_for(attempted.wait(), timeout=1)

    assert len(chunks) == 1
    response = chunks[0]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert isinstance(response.handling, ModelToolDetachedHandling)
    assert response.result.to_str() == "queued"


@pytest.mark.asyncio
async def test_handle_output_tool_yields_event_then_output_parts_then_response() -> None:
    async with ctx.scope("test"):

        @tool(handling="output")
        async def amplify(value: str):
            yield ProcessingEvent.of("progress", "starting")
            yield "OUT:"
            yield value

        chunks = [
            chunk
            async for chunk in Toolbox.of(amplify).handle(
                ModelToolRequest.of("r1", tool="amplify", arguments={"value": "x"})
            )
        ]

    assert len(chunks) == 4
    event = chunks[0]
    output_parts = chunks[1:3]
    response = chunks[3]
    assert isinstance(event, ProcessingEvent)
    assert event.event == "progress"
    assert event.meta["tool"] == "amplify"
    assert event.meta["identifier"] == "r1"
    assert _multimodal_text_of(*output_parts) == "OUT:x"
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert response.handling == "output"
    assert response.result.to_str() == "OUT:x"


@pytest.mark.asyncio
async def test_handle_output_tool_returns_error_response_with_partial_result() -> None:
    async with ctx.scope("test"):

        @tool(handling="output")
        async def unstable_output():
            yield "OUT:"
            raise RuntimeError("boom")

        chunks = [
            chunk
            async for chunk in Toolbox.of(unstable_output).handle(
                ModelToolRequest.of("r1", tool="unstable_output", arguments={})
            )
        ]

    assert len(chunks) == 2
    assert _multimodal_text_of(*chunks[:1]) == "OUT:"
    response = chunks[1]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "error"
    assert response.handling == "output"
    assert response.result.to_str() == "OUT:"


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

        model_tools = toolbox.with_tools(updated).model_tools()

    assert model_tools.selection == updated.specification


@pytest.mark.asyncio
async def test_tools_provider_toolbox_accepts_tool_suggestion() -> None:
    async with ctx.scope("test"):

        @tool
        async def ping() -> str:
            return "pong"

        toolbox = await ToolsProvider().toolbox(ping, suggesting=ping)

    assert toolbox.model_tools().selection == ping.specification


@pytest.mark.asyncio
async def test_tool_updating_allows_clearing_meta() -> None:
    async with ctx.scope("test"):

        @tool(meta={"tag": "value"})
        async def ping() -> str:
            return "pong"

        updated = ping.updating(meta=Meta.empty)

    assert ping.meta == Meta.of({"tag": "value"})
    assert updated.meta == Meta.empty
