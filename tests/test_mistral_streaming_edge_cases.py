from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from haiway import ctx
from mistralai.client.errors import SDKError
from mistralai.client.models import FunctionCall, ToolCall

from draive.mistral.completions import MistralCompletions
from draive.mistral.config import MistralChatConfig
from draive.models import (
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelOutputLimit,
    ModelToolRequest,
    ModelTools,
)
from draive.multimodal import TextContent


class _EventStream:
    def __init__(self, items: Sequence[Any], failure: Exception | None = None) -> None:
        self._items = items
        self._failure = failure
        self.closed: bool = False

    async def __aenter__(self) -> _EventStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _iterator() -> AsyncIterator[Any]:
            for item in self._items:
                yield item

            if self._failure is not None:
                raise self._failure

        return _iterator()


def _call(
    identifier: str | None = None,
    name: str = "",
    arguments: str | dict[str, Any] = "",
    index: int = 0,
) -> ToolCall:
    function = FunctionCall(name=name, arguments=arguments)
    if identifier is None:
        return ToolCall(function=function, index=index)

    return ToolCall(id=identifier, function=function, index=index)


def _event(
    content: Any = None,
    calls: Sequence[ToolCall] | None = None,
    finish: str | None = None,
) -> Any:
    return SimpleNamespace(
        data=SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=calls),
                    finish_reason=finish,
                )
            ],
            model="mistral-test",
        )
    )


async def _collect(
    events: Sequence[Any],
    *,
    failure: Exception | None = None,
    max_output_tokens: int | None = None,
) -> tuple[list[Any], _EventStream]:
    event_stream = _EventStream(events, failure)

    async def _stream_async(**_: Any) -> _EventStream:
        return event_stream

    model = object.__new__(MistralCompletions)
    model._client = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))  # pyright: ignore[reportPrivateUsage]
    chunks: list[Any] = []
    async with ctx.scope("test"):
        config = (
            MistralChatConfig(model="mistral-test", max_output_tokens=max_output_tokens)
            if max_output_tokens is not None
            else MistralChatConfig(model="mistral-test")
        )
        try:
            async for chunk in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=config,
            ):
                chunks.append(chunk)

        finally:
            assert event_stream.closed

    return chunks, event_stream


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["[1, 2]", '"text"', "3"])
async def test_non_object_tool_arguments_are_an_invalid_output(arguments: str) -> None:
    with pytest.raises(ModelOutputInvalid):
        await _collect([_event(calls=[_call("call-1", "echo", arguments)])])


@pytest.mark.asyncio
async def test_partial_tool_arguments_are_an_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid):
        await _collect([_event(calls=[_call("call-1", "echo", '{"x":')])])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [("", {}), ("  ", {}), ("null", {}), ("{}", {})],
)
async def test_blank_tool_arguments_are_an_empty_object(
    arguments: str,
    expected: dict[str, Any],
) -> None:
    chunks, _ = await _collect([_event(calls=[_call("call-1", "echo", arguments)])])

    assert [chunk.arguments for chunk in chunks] == [expected]


@pytest.mark.asyncio
async def test_continuation_with_the_default_identifier_extends_the_same_call() -> None:
    chunks, _ = await _collect(
        [
            _event(calls=[_call("call-1", "echo", '{"x"')]),
            _event(calls=[_call("null", "", ":1}")]),
            _event(finish="tool_calls"),
        ]
    )

    assert [(chunk.identifier, chunk.tool, chunk.arguments) for chunk in chunks] == [
        ("call-1", "echo", {"x": 1})
    ]


@pytest.mark.asyncio
async def test_parallel_calls_on_distinct_indexes_are_emitted_in_order() -> None:
    chunks, _ = await _collect(
        [
            _event(calls=[_call("call-1", "a", '{"x"', index=0)]),
            _event(calls=[_call(None, "", ":1}", index=0), _call("call-2", "b", "{}", index=1)]),
        ]
    )

    assert [(chunk.identifier, chunk.tool, chunk.arguments) for chunk in chunks] == [
        ("call-1", "a", {"x": 1}),
        ("call-2", "b", {}),
    ]


@pytest.mark.asyncio
async def test_text_following_a_tool_call_flushes_it_first() -> None:
    chunks, _ = await _collect(
        [
            _event(calls=[_call("call-1", "echo", "{}")]),
            _event(content="after"),
        ]
    )

    assert isinstance(chunks[0], ModelToolRequest)
    assert isinstance(chunks[1], TextContent)


@pytest.mark.asyncio
async def test_empty_content_and_unknown_finish_reason_are_tolerated() -> None:
    chunks, _ = await _collect(
        [_event(content=""), _event(content=[]), _event(content="x", finish="unexpected")]
    )

    assert [chunk.text for chunk in chunks] == ["x"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "model_length"])
async def test_length_finish_reasons_report_the_output_limit(reason: str) -> None:
    with pytest.raises(ModelOutputLimit) as caught:
        await _collect([_event(content="x", finish=reason)], max_output_tokens=7)

    assert caught.value.max_output_tokens == 7


@pytest.mark.asyncio
async def test_error_finish_reason_is_an_output_failure() -> None:
    with pytest.raises(ModelOutputFailed):
        await _collect([_event(content="x", finish="error")])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("refused"),
        SDKError("unavailable", httpx.Response(503, text="body")),
    ],
)
async def test_transport_and_server_failures_are_output_failures(failure: Exception) -> None:
    with pytest.raises(ModelOutputFailed) as caught:
        await _collect([_event(content="x")], failure=failure)

    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
async def test_stream_is_released_when_the_consumer_stops_early() -> None:
    event_stream = _EventStream([_event(content="a"), _event(content="b")])

    async def _stream_async(**_: Any) -> _EventStream:
        return event_stream

    model = object.__new__(MistralCompletions)
    model._client = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))  # pyright: ignore[reportPrivateUsage]
    async with ctx.scope("test"):
        stream = model.completion(
            instructions="",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=MistralChatConfig(model="mistral-test"),
        )
        async for _ in stream:
            break

        await stream.aclose()

    assert event_stream.closed
