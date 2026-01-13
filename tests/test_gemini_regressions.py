from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from google.genai.client import Client
from google.genai.errors import ClientError
from google.genai.types import FunctionCall, Part
from haiway import ctx

from draive.gemini.client import Gemini
from draive.gemini.config import GeminiConfig
from draive.gemini.generating import (
    GeminiGenerating,
    _part_as_stream_elements,
    _request_content,
)
from draive.models import (
    ModelOutput,
    ModelOutputFailed,
    ModelRateLimit,
    ModelToolRequest,
    ModelTools,
)


class _GeminiGeneratingForTest(GeminiGenerating):
    def __init__(self, client: object) -> None:
        self._client = cast(Client, client)


@pytest.mark.asyncio
async def test_gemini_stream_maps_client_error_429_to_model_rate_limit() -> None:
    async def _generate_content_stream(**_: Any) -> AsyncIterator[Any]:
        raise ClientError(
            code=429,
            response_json={"error": {"message": "rate limit"}},
            response=None,
        )

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_generate_content_stream,
            )
        )
    )

    model = _GeminiGeneratingForTest(fake_client)

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            context=(),
            tools=ModelTools.none,
            output="text",
            config=GeminiConfig(model="gemini-test", max_output_tokens=64),
        )
        with pytest.raises(ModelRateLimit):
            _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_gemini_stream_maps_non_429_client_error_to_model_output_failed() -> None:
    async def _generate_content_stream(**_: Any) -> AsyncIterator[Any]:
        raise ClientError(
            code=400,
            response_json={"error": {"message": "bad request"}},
            response=None,
        )

    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=_generate_content_stream,
            )
        )
    )

    model = _GeminiGeneratingForTest(fake_client)

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            context=(),
            tools=ModelTools.none,
            output="text",
            config=GeminiConfig(model="gemini-test", max_output_tokens=64),
        )
        with pytest.raises(ModelOutputFailed):
            _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_gemini_client_aexit_closes_sync_and_async_client() -> None:
    async_close = AsyncMock()
    sync_close = Mock()
    client: Any = object.__new__(Gemini)
    client._client = cast(
        Client,
        SimpleNamespace(
            aio=SimpleNamespace(aclose=async_close),
            close=sync_close,
        ),
    )

    await client.__aexit__(None, None, None)

    async_close.assert_awaited_once()
    sync_close.assert_called_once()


def test_gemini_tool_request_roundtrips_thought_signature() -> None:
    part = Part(
        function_call=FunctionCall(id="call-1", name="lookup", args={"query": "x"}),
        thought=True,
        thought_signature=b"sig",
    )

    elements = list(_part_as_stream_elements(part))

    assert len(elements) == 1
    tool_request = elements[0]
    assert isinstance(tool_request, ModelToolRequest)
    assert tool_request.meta.get_str("signature") == "c2ln"

    request_content = list(_request_content((ModelOutput.of(tool_request),)))

    assert request_content == [
        {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "id": "call-1",
                        "name": "lookup",
                        "args": {"query": "x"},
                    },
                    "thought_signature": b"sig",
                }
            ],
        }
    ]
