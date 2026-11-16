from types import SimpleNamespace
from typing import Any

import pytest
from haiway import ctx

from draive.models import (
    ModelInput,
    ModelOutput,
    ModelOutputFailed,
    ModelOutputInvalid,
    ModelSessionEvent,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent
from tests.test_openai_realtime import (
    _MockConnection,
    _prepared_session,
)


def _message(
    identifier: str | None,
    role: str,
    status: str,
    content: Any,
) -> Any:
    return SimpleNamespace(
        id=identifier,
        type="message",
        role=role,
        status=status,
        content=content,
    )


def _function_call(
    *,
    arguments: str | None,
    status: str = "completed",
) -> Any:
    return SimpleNamespace(
        type="response.output_item.done",
        response_id="resp-1",
        output_index=0,
        item=SimpleNamespace(
            id="item-1",
            type="function_call",
            status=status,
            call_id="call-1",
            name="echo",
            arguments=arguments,
        ),
    )


_TEXT_CONTENT: Any = [SimpleNamespace(type="output_text", text="hi", transcript=None, audio=None)]


async def _read(
    events: list[Any],
    *,
    reads: int,
) -> tuple[list[Any], _MockConnection, list[str]]:
    connection = _MockConnection()
    retrieved: list[str] = []

    async def _retrieve(*, item_id: str) -> None:
        retrieved.append(item_id)

    connection.conversation.item.retrieve = _retrieve  # pyright: ignore[reportAttributeAccessIssue]
    connection._events.extend(events)
    scope = await _prepared_session(connection)
    results: list[Any] = []
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        try:
            for _ in range(reads):
                results.append(await session._reading())

        finally:
            await scope.__aexit__(None, None, None)

    return results, connection, retrieved


@pytest.mark.asyncio
async def test_audio_and_text_deltas_become_chunks() -> None:
    results, _, _ = await _read(
        [
            SimpleNamespace(
                type="response.output_audio.delta",
                delta="AAEC",
                item_id="item-1",
                response_id="resp-1",
                output_index=0,
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="hello",
                item_id="item-1",
                response_id="resp-1",
                output_index=0,
            ),
        ],
        reads=2,
    )

    audio, text = results
    assert isinstance(audio, ResourceContent)
    assert audio.to_bytes() == b"\x00\x01\x02"
    assert audio.meta.get_str("item_id") == "item-1"
    assert isinstance(text, TextContent)
    assert text.text == "hello"


@pytest.mark.asyncio
async def test_malformed_audio_delta_is_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid):
        await _read(
            [
                SimpleNamespace(
                    type="response.output_audio.delta",
                    delta="not base64!",
                    item_id="item-1",
                    response_id="resp-1",
                    output_index=0,
                ),
            ],
            reads=1,
        )


@pytest.mark.asyncio
async def test_completed_function_call_becomes_tool_request() -> None:
    results, _, _ = await _read([_function_call(arguments='{"x": 1}')], reads=1)

    request = results[0]
    assert isinstance(request, ModelToolRequest)
    assert request.identifier == "call-1"
    assert request.tool == "echo"
    assert request.arguments == {"x": 1}


@pytest.mark.asyncio
async def test_function_call_without_arguments_has_empty_arguments() -> None:
    results, _, _ = await _read([_function_call(arguments="")], reads=1)

    assert results[0].arguments == {}


@pytest.mark.asyncio
async def test_unfinished_function_call_is_skipped() -> None:
    results, _, _ = await _read(
        [
            _function_call(arguments='{"x":', status="incomplete"),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="after",
                item_id="item-2",
                response_id="resp-1",
                output_index=1,
            ),
        ],
        reads=1,
    )

    assert isinstance(results[0], TextContent)


@pytest.mark.asyncio
async def test_malformed_function_call_arguments_are_invalid_output() -> None:
    with pytest.raises(ModelOutputInvalid):
        await _read([_function_call(arguments='{"x":')], reads=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ("[1]", '"text"'))
async def test_non_object_function_call_arguments_are_invalid_output(arguments: str) -> None:
    with pytest.raises(ModelOutputInvalid):
        await _read([_function_call(arguments=arguments)], reads=1)


@pytest.mark.asyncio
async def test_speech_events_map_to_turn_events() -> None:
    results, _, _ = await _read(
        [
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-1"),
            SimpleNamespace(type="input_audio_buffer.committed", item_id="item-1"),
        ],
        reads=2,
    )

    assert [result.event for result in results] == ["turn_started", "turn_commited"]


@pytest.mark.asyncio
async def test_retrieved_user_item_completes_the_input_turn() -> None:
    results, _, _ = await _read(
        [
            SimpleNamespace(
                type="conversation.item.retrieved",
                item=_message(
                    "item-1",
                    "user",
                    "completed",
                    [SimpleNamespace(type="input_text", text="question", transcript=None)],
                ),
            )
        ],
        reads=1,
    )

    event = results[0]
    assert isinstance(event, ModelSessionEvent)
    assert event.event == "turn_completed"
    assert isinstance(event.content, ModelInput)
    assert event.content.content.to_str() == "question"


@pytest.mark.asyncio
async def test_retrieved_unfinished_item_is_requested_again() -> None:
    results, _, retrieved = await _read(
        [
            SimpleNamespace(
                type="conversation.item.retrieved",
                item=_message("item-1", "assistant", "in_progress", []),
            ),
            SimpleNamespace(
                type="conversation.item.retrieved",
                item=_message("item-1", "assistant", "completed", _TEXT_CONTENT),
            ),
        ],
        reads=1,
    )

    assert retrieved == ["item-1"]
    assert isinstance(results[0].content, ModelOutput)


@pytest.mark.asyncio
async def test_usage_is_recorded_on_response_done(monkeypatch: pytest.MonkeyPatch) -> None:
    usage: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "draive.openai.realtime.record_usage_metrics",
        lambda **kwargs: usage.append(kwargs),
    )
    await _read(
        [
            SimpleNamespace(
                type="response.done",
                response=SimpleNamespace(
                    id="resp-1",
                    usage=SimpleNamespace(
                        input_tokens=4,
                        input_token_details=SimpleNamespace(cached_tokens=1),
                        output_tokens=6,
                    ),
                    status="completed",
                    status_details=None,
                ),
            ),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-1"),
        ],
        reads=1,
    )

    assert usage == [
        {
            "provider": "openai",
            "model": "gpt-realtime",
            "input_tokens": 4,
            "cached_input_tokens": 1,
            "output_tokens": 6,
        }
    ]


@pytest.mark.asyncio
async def test_committed_input_buffer_is_not_committed_again() -> None:
    connection = _MockConnection()
    connection._events.append(
        SimpleNamespace(type="input_audio_buffer.committed", item_id="item-1")
    )
    scope = await _prepared_session(connection)
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        await session._writing(ResourceContent.of(b"\x00\x01", mime_type="audio/pcm"))
        # server side voice detection committed the buffer on its own
        await session._reading()
        await session._writing(ModelSessionEvent.turn_commited())
        await scope.__aexit__(None, None, None)

    assert connection.input_audio_buffer.commit_calls == 0
    assert connection.response.create_calls == 1


@pytest.mark.asyncio
async def test_tool_response_is_delivered_and_requests_a_response() -> None:
    connection = _MockConnection()
    scope = await _prepared_session(connection)
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        await session._writing(
            ModelToolResponse.of("call-1", tool="echo", content=MultimodalContent.of("done"))
        )
        await scope.__aexit__(None, None, None)

    item = connection.conversation.item.created_items[0]
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call-1"
    assert item["output"] == "done"
    assert connection.response.create_calls == 1


@pytest.mark.asyncio
async def test_failed_connection_releases_the_scope() -> None:
    connection = _MockConnection()
    scope = await _prepared_session(connection)

    async def _failing_update(*, session: Any) -> None:
        raise RuntimeError("update failed")

    connection.session.update = _failing_update  # pyright: ignore[reportAttributeAccessIssue]
    async with ctx.scope("test"):
        with pytest.raises(RuntimeError, match="update failed"):
            await scope.__aenter__()


@pytest.mark.asyncio
async def test_assistant_item_is_completed_once() -> None:
    results, _, _ = await _read(
        [
            SimpleNamespace(
                type="conversation.item.done",
                item=_message("item-1", "assistant", "completed", _TEXT_CONTENT),
            ),
            SimpleNamespace(
                type="conversation.item.retrieved",
                item=_message("item-1", "assistant", "completed", _TEXT_CONTENT),
            ),
            SimpleNamespace(
                type="conversation.item.retrieved",
                item=_message("item-1", "assistant", "completed", _TEXT_CONTENT),
            ),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-2"),
        ],
        reads=2,
    )

    assert [result.event for result in results].count("turn_completed") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ([], None))
async def test_completed_item_without_content_does_not_break_the_session(content: Any) -> None:
    results, _, _ = await _read(
        [
            SimpleNamespace(
                type="conversation.item.retrieved",
                item=_message("item-1", "assistant", "completed", content),
            ),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-2"),
        ],
        reads=1,
    )

    assert results[0].event == "turn_started"


@pytest.mark.asyncio
async def test_error_event_is_reported_as_output_failure() -> None:
    with pytest.raises(ModelOutputFailed, match="bad request"):
        await _read(
            [
                SimpleNamespace(
                    type="error",
                    error=SimpleNamespace(
                        type="invalid_request_error", code=None, message="bad request"
                    ),
                )
            ],
            reads=1,
        )


def _response_done(response_id: str = "resp-1") -> Any:
    return SimpleNamespace(
        type="response.done",
        response=SimpleNamespace(
            id=response_id,
            usage=None,
            status="completed",
            status_details=None,
        ),
    )


def _named_function_call(call_id: str) -> Any:
    event = _function_call(arguments="{}")
    event.item.id = f"item-{call_id}"
    event.item.call_id = call_id
    return event


@pytest.mark.asyncio
async def test_parallel_tool_responses_request_a_single_response() -> None:
    connection = _MockConnection()
    connection._events.extend(
        [
            _named_function_call("call-a"),
            _named_function_call("call-b"),
            _response_done(),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-1"),
        ]
    )
    scope = await _prepared_session(connection)
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        await session._reading()
        await session._reading()
        await session._reading()  # response completed
        await session._writing(
            ModelToolResponse.of("call-a", tool="echo", content=MultimodalContent.of("a"))
        )
        # the other call is still unanswered, the model would see it without output
        assert connection.response.create_calls == 0
        await session._writing(
            ModelToolResponse.of("call-b", tool="echo", content=MultimodalContent.of("b"))
        )
        await scope.__aexit__(None, None, None)

    assert [item["call_id"] for item in connection.conversation.item.created_items] == [
        "call-a",
        "call-b",
    ]
    assert connection.response.create_calls == 1


@pytest.mark.asyncio
async def test_tool_answered_before_next_call_arrives_waits_for_the_response() -> None:
    connection = _MockConnection()
    connection._events.extend(
        [
            _named_function_call("call-a"),
            _named_function_call("call-b"),
            _response_done(),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-1"),
        ]
    )
    scope = await _prepared_session(connection)
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        await session._reading()  # call-a requested
        await session._writing(
            ModelToolResponse.of("call-a", tool="echo", content=MultimodalContent.of("a"))
        )
        # call-b was not reported yet and the response is still being produced
        assert connection.response.create_calls == 0
        await session._reading()  # call-b requested
        await session._reading()  # response done, call-b is still unanswered
        assert connection.response.create_calls == 0
        await session._writing(
            ModelToolResponse.of("call-b", tool="echo", content=MultimodalContent.of("b"))
        )
        await scope.__aexit__(None, None, None)

    assert connection.response.create_calls == 1


@pytest.mark.asyncio
async def test_tool_answered_before_response_done_requests_response_when_done() -> None:
    connection = _MockConnection()
    connection._events.extend(
        [
            _named_function_call("call-a"),
            _response_done(),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-1"),
        ]
    )
    scope = await _prepared_session(connection)
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        await session._reading()
        await session._writing(
            ModelToolResponse.of("call-a", tool="echo", content=MultimodalContent.of("a"))
        )
        assert connection.response.create_calls == 0
        await session._reading()  # response done
        assert connection.response.create_calls == 1
        await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_tool_answered_after_response_done_requests_response_immediately() -> None:
    connection = _MockConnection()
    connection._events.extend(
        [
            _named_function_call("call-a"),
            _response_done(),
            SimpleNamespace(type="input_audio_buffer.speech_started", item_id="item-1"),
        ]
    )
    scope = await _prepared_session(connection)
    async with ctx.scope("test"):
        session = await scope.__aenter__()
        await session._reading()
        await session._reading()
        assert connection.response.create_calls == 0
        await session._writing(
            ModelToolResponse.of("call-a", tool="echo", content=MultimodalContent.of("a"))
        )
        assert connection.response.create_calls == 1
        await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    ("input_audio_buffer_commit_empty", "response_cancel_not_active"),
)
async def test_benign_error_events_do_not_end_the_session(code: str) -> None:
    results, _, _ = await _read(
        [
            SimpleNamespace(
                type="error",
                error=SimpleNamespace(type="invalid_request_error", code=code, message="benign"),
            ),
            _function_call(arguments="{}"),
        ],
        reads=1,
    )

    assert isinstance(results[0], ModelToolRequest)


@pytest.mark.asyncio
async def test_tool_only_turns_do_not_create_empty_items() -> None:
    from draive.openai.realtime import _send_context  # pyright: ignore[reportPrivateUsage]

    connection = _MockConnection()
    await _send_context(
        (
            ModelInput.of(MultimodalContent.of("go")),
            ModelOutput.of(ModelToolRequest.of("call-1", tool="echo", arguments={})),
            ModelInput.of(
                ModelToolResponse.of("call-1", tool="echo", content=MultimodalContent.of("done"))
            ),
        ),
        current_items={},
        connection=connection,  # pyright: ignore[reportArgumentType]
    )

    items = connection.conversation.item.created_items
    assert [item["type"] for item in items] == ["message", "function_call", "function_call_output"]
