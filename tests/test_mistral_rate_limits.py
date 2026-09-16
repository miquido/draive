from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from haiway import ctx
from mistralai.client.errors import SDKError

from draive.mistral.completions import MistralCompletions
from draive.mistral.config import MistralChatConfig
from draive.models import ModelQuotaLimit, ModelRateLimit, ModelTools


@pytest.mark.asyncio
@pytest.mark.parametrize("during_stream", [False, True])
@pytest.mark.parametrize("limit", ["0", " 0.0 "])
async def test_zero_capacity_is_not_retryable(limit: str, during_stream: bool) -> None:
    with pytest.raises(ModelQuotaLimit) as caught:
        await _complete({"x-ratelimit-limit-req-minute": limit}, during_stream=during_stream)

    assert not isinstance(caught.value, ModelRateLimit)
    assert caught.value.provider == "mistral"
    assert caught.value.model == "test-model"
    assert "sensitive body" not in str(caught.value)
    assert caught.value.__cause__ is not None
    assert "zero quota capacity" in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, "12", "invalid"])
async def test_burst_limit_preserves_retry_after(limit: str | None) -> None:
    headers = {"Retry-After": "12", "x-ratelimit-remaining-req-minute": "0"}
    if limit is not None:
        headers["x-ratelimit-limit-req-minute"] = limit

    with pytest.raises(ModelRateLimit) as caught:
        await _complete(headers)

    assert caught.value.retry_after == 12


async def _complete(headers: Mapping[str, str], *, during_stream: bool = False) -> None:
    error = SDKError("rate limited", httpx.Response(429, headers=headers, text="sensitive body"))

    class Stream:
        async def __aenter__(self) -> Any:
            raise error

        async def __aexit__(self, *_: Any) -> None:
            pass

    async def stream_async(**_: Any) -> Any:
        if during_stream:
            return Stream()
        raise error

    model = object.__new__(MistralCompletions)
    model._client = SimpleNamespace(chat=SimpleNamespace(stream_async=stream_async))
    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="test",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=MistralChatConfig(model="test-model"),
        ):
            pass
