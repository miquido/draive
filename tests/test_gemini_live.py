from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace, TracebackType
from typing import Any

import pytest
from google.genai.types import (
    Content,
    LiveServerContent,
    LiveServerGoAway,
    LiveServerMessage,
    Part,
)
from haiway import ctx

from draive.gemini.config import GeminiConfig
from draive.gemini.live import GeminiLive
from draive.models import (
    ModelInput,
    ModelSession,
    ModelSessionScope,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent


class _FakeSession:
    def __init__(
        self,
        messages: Sequence[LiveServerMessage] = (),
    ) -> None:
        self.client_content_calls: list[dict[str, Any]] = []
        self._messages: list[LiveServerMessage] = list(messages)

    async def send_client_content(
        self,
        *,
        turns: Any = None,
        turn_complete: bool = True,
    ) -> None:
        self.client_content_calls.append(
            {
                "turns": turns,
                "turn_complete": turn_complete,
            }
        )

    def receive(self) -> AsyncIterator[LiveServerMessage]:
        async def _iterator() -> AsyncIterator[LiveServerMessage]:
            while self._messages:
                yield self._messages.pop(0)

        return _iterator()


class _FakeConnectionManager:
    def __init__(
        self,
        session: _FakeSession,
    ) -> None:
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


def _prepare(
    session: _FakeSession,
    /,
    *,
    context: Any = (),
) -> ModelSessionScope:
    def _connect(**_: Any) -> Any:
        return _FakeConnectionManager(session)

    model = object.__new__(GeminiLive)
    model._client = SimpleNamespace(
        aio=SimpleNamespace(
            live=SimpleNamespace(connect=_connect),
        )
    )

    return model.session_prepare(
        instructions="",
        tools=ModelTools.none,
        context=context,
        output="text",
        config=GeminiConfig(model="gemini-live-test"),
    )


@pytest.mark.asyncio
async def test_gemini_live_skips_initial_turn_without_context() -> None:
    session = _FakeSession()
    scope: ModelSessionScope = _prepare(session)

    async with ctx.scope("test"), scope:
        pass

    # an empty client content turn is rejected by the api with an invalid argument error
    assert session.client_content_calls == []


@pytest.mark.asyncio
async def test_gemini_live_sends_initial_turn_with_context() -> None:
    session = _FakeSession()
    scope: ModelSessionScope = _prepare(
        session,
        context=(ModelInput.of(MultimodalContent.of("hello")),),
    )

    async with ctx.scope("test"), scope:
        pass

    assert len(session.client_content_calls) == 1
    call = session.client_content_calls[0]
    # completing the initial history turn is rejected by the api,
    # realtime input starts the turn instead
    assert call["turn_complete"] is False
    assert call["turns"] == [{"role": "user", "parts": [{"text": "hello"}]}]


@pytest.mark.asyncio
async def test_gemini_live_go_away_does_not_fail_session() -> None:
    session = _FakeSession(
        (
            LiveServerMessage(go_away=LiveServerGoAway(time_left="10s")),
            LiveServerMessage(
                server_content=LiveServerContent(
                    model_turn=Content(
                        role="model",
                        parts=[Part(text="continuing")],
                    ),
                ),
            ),
        )
    )
    scope: ModelSessionScope = _prepare(session)

    async with ctx.scope("test"):
        model_session: ModelSession
        async with scope as model_session, ctx.scope("session", model_session):
            chunk = await ModelSession.read()

    assert isinstance(chunk, TextContent)
    assert chunk.text == "continuing"
