from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace, TracebackType
from typing import Any

import pytest
from google.genai.errors import APIError
from google.genai.types import (
    Blob,
    Content,
    FunctionCall,
    LiveServerContent,
    LiveServerMessage,
    LiveServerToolCall,
    Part,
    TurnCompleteReason,
    UsageMetadata,
)
from haiway import ctx

from draive.gemini.config import GeminiConfig
from draive.gemini.live import GeminiLive
from draive.models import (
    ModelOutput,
    ModelOutputFailed,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelSession,
    ModelSessionEvent,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent


class _Session:
    def __init__(
        self,
        messages: Sequence[LiveServerMessage],
        *,
        failure: BaseException | None = None,
    ) -> None:
        self._messages: list[LiveServerMessage] = list(messages)
        self._failure: BaseException | None = failure
        self.realtime_calls: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []

    async def send_client_content(self, **_: Any) -> None:
        return None

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime_calls.append(kwargs)

    async def send_tool_response(self, *, function_responses: Any) -> None:
        self.tool_responses.append(function_responses)

    def receive(self) -> AsyncIterator[LiveServerMessage]:
        async def _iterator() -> AsyncIterator[LiveServerMessage]:
            while self._messages:
                yield self._messages.pop(0)

            # the real stream never ends silently, it waits or fails
            if self._failure is not None:
                raise self._failure

            raise AssertionError("the test read more events than it queued")

        return _iterator()


class _Manager:
    def __init__(self, session: _Session) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None


async def _open(session: _Session) -> tuple[ModelSession, Any]:
    model = object.__new__(GeminiLive)
    model._client = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        aio=SimpleNamespace(live=SimpleNamespace(connect=lambda **_: _Manager(session)))
    )
    scope = model.session_prepare(
        instructions="",
        tools=ModelTools.none,
        context=(),
        output="text",
        config=GeminiConfig(model="gemini-live-test"),
    )

    return await scope.__aenter__(), scope


def _content(**kwargs: Any) -> LiveServerMessage:
    return LiveServerMessage(server_content=LiveServerContent(**kwargs))


def _audio(data: bytes) -> Content:
    return Content(
        role="model",
        parts=[Part(inline_data=Blob(data=data, mime_type="audio/pcm;rate=24000"))],
    )


@pytest.mark.asyncio
async def test_model_turn_text_is_streamed_and_collected_into_the_completed_turn() -> None:
    session = _Session(
        [
            _content(model_turn=Content(role="model", parts=[Part(text="Hello")])),
            _content(generation_complete=True),
            _content(turn_complete=True, turn_complete_reason=TurnCompleteReason.NEED_MORE_INPUT),
        ]
    )
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        chunk = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        finished = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        completed = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert isinstance(chunk, TextContent)
    assert chunk.text == "Hello"
    assert isinstance(finished, ModelSessionEvent)
    assert finished.event == "turn_finished"
    assert isinstance(completed, ModelSessionEvent)
    assert completed.event == "turn_completed"
    assert isinstance(completed.content, ModelOutput)
    assert completed.content.content.to_str() == "Hello"
    assert completed.content.meta.get_str("reason") == "NEED_MORE_INPUT"


@pytest.mark.asyncio
async def test_output_transcription_joins_the_completed_turn() -> None:
    session = _Session(
        [
            _content(output_transcription={"text": "Hel"}),
            _content(output_transcription={"text": "lo"}),
            _content(turn_complete=True),
        ]
    )
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        completed = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert isinstance(completed, ModelSessionEvent)
    assert isinstance(completed.content, ModelOutput)
    assert completed.content.content.to_str() == "Hello"


@pytest.mark.asyncio
async def test_finished_input_transcription_completes_the_input_turn() -> None:
    session = _Session(
        [
            _content(input_transcription={"text": "Hi "}),
            _content(input_transcription={"text": "there", "finished": True}),
        ]
    )
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        completed = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert isinstance(completed, ModelSessionEvent)
    assert completed.event == "turn_completed"
    assert completed.content is not None
    assert not isinstance(completed.content, ModelOutput)
    assert completed.content.kind == "input"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_tool_call_message_becomes_tool_requests() -> None:
    session = _Session(
        [
            LiveServerMessage(
                tool_call=LiveServerToolCall(
                    function_calls=[
                        FunctionCall(id="call-1", name="echo", args={"x": 1}),
                        FunctionCall(id="call-2", name=None),
                        FunctionCall(id="call-3", name="ping", args=None),
                    ]
                )
            ),
        ]
    )
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        first = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        second = await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    # a call without a name is skipped
    assert isinstance(first, ModelToolRequest)
    assert (first.identifier, first.tool, first.arguments) == ("call-1", "echo", {"x": 1})
    assert isinstance(second, ModelToolRequest)
    assert (second.identifier, second.tool, second.arguments) == ("call-3", "ping", {})


@pytest.mark.asyncio
async def test_usage_metadata_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "draive.gemini.live.record_usage_metrics",
        lambda **kwargs: recorded.append(kwargs),  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType, reportUnknownArgumentType]
    )
    session = _Session(
        [
            LiveServerMessage(
                usage_metadata=UsageMetadata(
                    prompt_token_count=5,
                    response_token_count=7,
                    thoughts_token_count=2,
                ),
                server_content=LiveServerContent(
                    model_turn=Content(role="model", parts=[Part(text="x")])
                ),
            )
        ]
    )
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        await model_session._reading()  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert len(recorded) == 1
    assert recorded[0]["input_tokens"] == 5
    assert recorded[0]["output_tokens"] == 7
    assert recorded[0]["reasoning_output_tokens"] == 2


@pytest.mark.asyncio
async def test_rate_limit_close_is_translated() -> None:
    session = _Session([], failure=APIError(1011, {"message": "RESOURCE_EXHAUSTED: slow down"}))
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        with pytest.raises(ModelRateLimit):
            await model_session._reading()  # pyright: ignore[reportPrivateUsage]

        await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_zero_quota_close_is_not_retryable() -> None:
    error = APIError(
        429,
        {
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaValue": "0"}],
                }
            ]
        },
    )
    session = _Session([], failure=error)
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        with pytest.raises(ModelQuotaLimit):
            await model_session._reading()  # pyright: ignore[reportPrivateUsage]

        await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_other_api_errors_are_output_failures() -> None:
    session = _Session([], failure=APIError(1008, {"message": "policy violation"}))
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        with pytest.raises(ModelOutputFailed):
            await model_session._reading()  # pyright: ignore[reportPrivateUsage]

        await scope.__aexit__(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mime_type", "keyword"),
    (
        ("audio/pcm;rate=16000", "audio"),
        ("image/jpeg", "media"),
        ("video/mp4", "video"),
    ),
)
async def test_input_media_is_routed_by_modality(mime_type: str, keyword: str) -> None:
    session = _Session([])
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        await model_session._writing(ResourceContent.of(b"\x00\x01", mime_type=mime_type))  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert len(session.realtime_calls) == 1
    assert list(session.realtime_calls[0]) == [keyword]
    assert session.realtime_calls[0][keyword].mime_type == mime_type
    assert session.realtime_calls[0][keyword].data == b"\x00\x01"


@pytest.mark.asyncio
async def test_unsupported_input_media_is_skipped() -> None:
    session = _Session([])
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        await model_session._writing(ResourceContent.of(b"%PDF", mime_type="application/pdf"))  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert session.realtime_calls == []


@pytest.mark.asyncio
async def test_text_tool_response_and_activity_events_are_delivered() -> None:
    session = _Session([])
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        await model_session._writing(TextContent.of("hello"))  # pyright: ignore[reportPrivateUsage]
        await model_session._writing(  # pyright: ignore[reportPrivateUsage]
            ModelToolResponse.of("call-1", tool="echo", content=MultimodalContent.of("done"))
        )
        await model_session._writing(ModelSessionEvent.turn_started())  # pyright: ignore[reportPrivateUsage]
        await model_session._writing(ModelSessionEvent.turn_commited())  # pyright: ignore[reportPrivateUsage]
        await scope.__aexit__(None, None, None)

    assert session.realtime_calls == [
        {"text": "hello"},
        {"activity_start": {}},
        {"activity_end": {}},
    ]
    assert len(session.tool_responses) == 1
    assert session.tool_responses[0][0]["id"] == "call-1"
    assert session.tool_responses[0][0]["response"] == {"output": "done"}


@pytest.mark.asyncio
async def test_interrupted_turn_output_does_not_leak_into_the_next_turn() -> None:
    session = _Session(
        [
            _content(model_turn=_audio(b"old")),
            _content(output_transcription={"text": "stale "}),
            _content(interrupted=True),
            _content(model_turn=_audio(b"new")),
            _content(output_transcription={"text": "fresh"}),
            _content(turn_complete=True),
        ]
    )
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        events: list[Any] = [
            await model_session._reading()  # pyright: ignore[reportPrivateUsage]
            for _ in range(4)
        ]
        await scope.__aexit__(None, None, None)

    completed = events[-1]
    assert isinstance(completed, ModelSessionEvent)
    assert isinstance(completed.content, ModelOutput)
    assert "stale" not in completed.content.content.to_str()


@pytest.mark.asyncio
async def test_unavailable_close_is_retryable() -> None:
    session = _Session([], failure=APIError(1011, {"message": "UNAVAILABLE: overloaded"}))
    async with ctx.scope("test"):
        model_session, scope = await _open(session)
        with pytest.raises(ModelRateLimit):
            await model_session._reading()  # pyright: ignore[reportPrivateUsage]

        await scope.__aexit__(None, None, None)


def test_missing_function_call_identifier_is_shared_between_stream_and_output() -> None:
    from google.genai.types import FunctionCall, Part

    from draive.gemini.content import part_as_output_blocks, part_as_stream_elements
    from draive.models import ModelToolRequest

    part = Part(function_call=FunctionCall(name="echo", args={"x": 1}))

    streamed = [e for e in part_as_stream_elements(part) if isinstance(e, ModelToolRequest)]
    output = [b for b in part_as_output_blocks(part) if isinstance(b, ModelToolRequest)]

    assert len(streamed) == len(output) == 1
    assert streamed[0].identifier == output[0].identifier
    assert streamed[0].identifier
