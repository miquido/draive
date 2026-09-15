from collections.abc import AsyncIterator, Mapping
from types import SimpleNamespace
from typing import Any, Literal

import httpx
import pytest
from anthropic import RateLimitError as AnthropicRateLimitError
from haiway import ctx
from openai import RateLimitError as OpenAIRateLimitError

from draive.anthropic.config import AnthropicConfig
from draive.anthropic.messages import AnthropicMessages
from draive.models import ModelQuotaLimit, ModelRateLimit, ModelTools
from draive.openai.config import OpenAIResponsesConfig
from draive.openai.responses import OpenAIResponses
from draive.vllm.config import VLLMChatConfig
from draive.vllm.messages import VLLMMessages

type Provider = Literal["openai", "vllm", "anthropic"]

_CAPACITY_HEADERS: tuple[tuple[Provider, str], ...] = (
    ("openai", "X-RateLimit-Limit-Requests"),
    ("openai", "x-ratelimit-limit-tokens"),
    ("vllm", "x-ratelimit-limit-requests"),
    ("vllm", "x-ratelimit-limit-tokens"),
    ("anthropic", "anthropic-ratelimit-requests-limit"),
    ("anthropic", "anthropic-ratelimit-tokens-limit"),
    ("anthropic", "anthropic-ratelimit-input-tokens-limit"),
    ("anthropic", "anthropic-ratelimit-output-tokens-limit"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, header", _CAPACITY_HEADERS)
@pytest.mark.parametrize("value", ["0", " 0.0 "])
@pytest.mark.parametrize("during_stream", [False, True])
async def test_zero_capacity_overrides_retry_delay(
    provider: Provider, header: str, value: str, during_stream: bool
) -> None:
    with pytest.raises(ModelQuotaLimit) as caught:
        await _complete(provider, {header: value, "Retry-After": "15"}, during_stream=during_stream)

    assert caught.value.provider == ("vllm@https://test" if provider == "vllm" else provider)
    assert caught.value.model == "test-model"
    assert caught.value.__cause__ is not None
    assert header not in str(caught.value)
    assert "sensitive body" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, header", _CAPACITY_HEADERS)
@pytest.mark.parametrize("value", ["100", "invalid", "NaN", "Infinity"])
async def test_nonzero_or_invalid_capacity_preserves_retry_delay(
    provider: Provider, header: str, value: str
) -> None:
    with pytest.raises(ModelRateLimit) as caught:
        await _complete(provider, {header: value, "Retry-After": "15"})

    assert caught.value.retry_after == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "vllm", "anthropic"])
@pytest.mark.parametrize("delay", [None, "invalid"])
async def test_remaining_capacity_and_unknown_headers_do_not_establish_quota(
    provider: Provider, delay: str | None
) -> None:
    headers = {
        "x-ratelimit-remaining-requests": "0",
        "anthropic-ratelimit-requests-remaining": "0",
        "unrecognized-limit": "0",
    }
    if delay is not None:
        headers["Retry-After"] = delay
    with pytest.raises(ModelRateLimit) as caught:
        await _complete(provider, headers)

    assert 0.3 <= caught.value.retry_after <= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "vllm"])
@pytest.mark.parametrize("code", ["insufficient_quota", "billing_hard_limit_reached"])
async def test_error_code_establishes_quota(provider: Provider, code: str) -> None:
    with pytest.raises(ModelQuotaLimit):
        await _complete(provider, {"Retry-After": "15"}, code=code)


async def _complete(
    provider: Provider,
    headers: Mapping[str, str],
    *,
    during_stream: bool = False,
    code: str | None = None,
) -> None:
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://test"))
    error = (
        AnthropicRateLimitError("sensitive body", response=response, body={})
        if provider == "anthropic"
        else OpenAIRateLimitError("sensitive body", response=response, body={"code": code})
    )

    def stream(**_: Any) -> _ErrorStream:
        if during_stream:
            return _ErrorStream(error)
        raise error

    async def create(**kwargs: Any) -> _ErrorStream:
        return stream(**kwargs)

    # The fake SDK clients model only the provider boundary exercised here.
    model: Any
    config: OpenAIResponsesConfig | VLLMChatConfig | AnthropicConfig
    match provider:
        case "openai":
            model = object.__new__(OpenAIResponses)
            model._base_url = None
            model._client = SimpleNamespace(responses=SimpleNamespace(stream=stream))
            config = OpenAIResponsesConfig(model="test-model")
        case "vllm":
            model = object.__new__(VLLMMessages)
            model._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            model._base_url = "https://test"
            config = VLLMChatConfig(model="test-model")
        case "anthropic":
            model = object.__new__(AnthropicMessages)
            model._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
            model._provider = provider
            config = AnthropicConfig(model="test-model")

    async with ctx.scope("test"):
        async for _ in model.completion(
            instructions="test",
            tools=ModelTools.none,
            context=(),
            output="text",
            config=config,
        ):
            pass


class _ErrorStream:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> _ErrorStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        raise self._error
