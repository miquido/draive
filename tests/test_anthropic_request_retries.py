import json
from collections.abc import Sequence
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from draive.anthropic.messages import _context_messages  # pyright: ignore[reportPrivateUsage]
from draive.models import ModelInput, ModelToolResponse
from draive.multimodal import MultimodalContent, TextContent

_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.mark.asyncio
async def test_tool_result_content_survives_request_retries() -> None:
    messages: Sequence[MessageParam] = list(
        _context_messages(
            (
                ModelInput.of(
                    ModelToolResponse.of(
                        "c",
                        tool="t",
                        content=MultimodalContent.of(TextContent.of("result")),
                    ),
                    MultimodalContent.of(TextContent.of("question")),
                ),
            )
        )
    )
    bodies: list[Any] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error"}})

        return httpx2.Response(200, json=_MESSAGE)

    async with AsyncAnthropic(
        api_key="key",
        max_retries=1,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        await client.messages.create(model="m", max_tokens=1, messages=messages)

    assert len(bodies) == 2
    assert bodies[0] == bodies[1]
    assert bodies[1]["messages"][0]["content"][0]["content"] == [{"type": "text", "text": "result"}]
