import asyncio

import pytest
from haiway import ctx

from draive.models import ModelOutputFailed, ModelRateLimit, provider_failure, provider_gather
from draive.utils import gather_all


@pytest.mark.asyncio
async def test_provider_failure_translates_unexpected_errors() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed, match="embedding request failed - RuntimeError"):
            with provider_failure(provider="test", model="m", operation="embedding"):
                raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_provider_failure_passes_model_exceptions_through() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit):
            with provider_failure(provider="test", model="m", operation="embedding"):
                raise ModelRateLimit(provider="test", model="m", retry_after=1.0)


@pytest.mark.asyncio
async def test_gather_all_cancels_siblings_and_raises_original() -> None:
    cancelled = asyncio.Event()

    async def slow() -> int:
        try:
            await asyncio.sleep(10)
            return 1

        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def failing() -> int:
        await asyncio.sleep(0)
        raise ValueError("first")

    with pytest.raises(ValueError, match="first"):
        await gather_all(slow(), failing())

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_gather_all_preserves_order() -> None:
    async def value(number: int) -> int:
        await asyncio.sleep(0.01 * (3 - number))
        return number

    assert await gather_all(value(1), value(2), value(3)) == (1, 2, 3)


@pytest.mark.asyncio
async def test_provider_gather_translates_failures() -> None:
    async def failing() -> int:
        raise RuntimeError("boom")

    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed):
            await provider_gather(failing(), provider="test", model="m", operation="embedding")


class _StatusCodeError(Exception):
    def __init__(self, status_code: int, headers: object = None) -> None:
        super().__init__("failed")
        self.status_code = status_code
        self.headers = headers


class _CodeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__("failed")
        self.code = code


class _ResponseError(Exception):
    def __init__(self, status_code: int, headers: object) -> None:
        super().__init__("failed")
        self.response = type("Response", (), {"headers": headers})()
        self.status_code = status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "retry_after"),
    (
        (_StatusCodeError(429, {"Retry-After": "9"}), 9.0),  # cohere style plain mapping
        (_ResponseError(429, {"retry-after": "4"}), 4.0),  # openai/anthropic style response
        (_CodeError(429), None),  # gemini style, no delay known
        (_StatusCodeError(429), None),  # ollama style
    ),
)
async def test_provider_failure_maps_rate_limit_status(
    error: Exception, retry_after: float | None
) -> None:
    async with ctx.scope("test"):
        with pytest.raises(ModelRateLimit) as caught:
            with provider_failure(provider="test", model="m", operation="embedding"):
                raise error

    if retry_after is not None:
        assert caught.value.retry_after == retry_after

    else:
        assert 0.3 <= caught.value.retry_after <= 3  # randomized fallback


@pytest.mark.asyncio
async def test_provider_failure_keeps_other_statuses_as_failures() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ModelOutputFailed):
            with provider_failure(provider="test", model="m", operation="embedding"):
                raise _StatusCodeError(500)


def test_mistral_image_chunks_infer_mime_type_from_url_extension() -> None:
    from mistralai.client.models import ImageURLChunk

    from draive.mistral.completions import (
        _content_chunk_as_content_element,  # pyright: ignore[reportPrivateUsage]
    )

    def mime_type(url: str) -> str:
        element = _content_chunk_as_content_element(ImageURLChunk(image_url=url))
        assert element is not None
        return element.mime_type  # pyright: ignore[reportAttributeAccessIssue]

    assert mime_type("https://example.com/a/photo.jpg?token=1") == "image/jpeg"
    assert mime_type("https://example.com/a/diagram.webp") == "image/webp"
    assert mime_type("https://example.com/a/unknown") == "image/png"


@pytest.mark.asyncio
async def test_text_only_providers_warn_when_dropping_tool_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from draive.models import ModelInput, ModelToolResponse
    from draive.multimodal import MultimodalContent, TextContent
    from draive.ollama.chat import _context_messages as ollama_messages  # pyright: ignore[reportPrivateUsage]
    from draive.resources import ResourceReference
    from draive.vllm.messages import _context_messages as vllm_messages  # pyright: ignore[reportPrivateUsage]

    warnings: list[str] = []
    monkeypatch.setattr(ctx, "log_warning", lambda message, *_, **__: warnings.append(message))

    context = (
        ModelInput.of(
            MultimodalContent.of("hi"),
            ModelToolResponse(
                identifier="call",
                tool="screenshot",
                status="success",
                content=MultimodalContent.of(
                    TextContent.of("shot"),
                    ResourceReference.of("https://example.com/s.png", mime_type="image/png"),
                ),
            ),
        ),
    )

    async with ctx.scope("test"):
        list(ollama_messages(instructions="", context=context))
        for message in vllm_messages(instructions="", context=context, vision_details="auto"):
            if message["role"] == "tool":
                list(message["content"])  # type: ignore[arg-type]

    assert any("Ollama" in warning and "screenshot" in warning for warning in warnings)
    assert any("vLLM" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_mistral_closes_sync_client_even_when_async_close_fails() -> None:
    from draive.mistral.api import MistralAPI

    closed: list[str] = []

    class _Client:
        async def __aexit__(self, *_: object) -> None:
            raise RuntimeError("async close failed")

        def __exit__(self, *_: object) -> None:
            closed.append("sync")

    api = object.__new__(MistralAPI)
    api._client = _Client()  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(RuntimeError, match="async close failed"):
        await api._deinitialize_client()  # pyright: ignore[reportPrivateUsage]

    assert closed == ["sync"]
    assert not hasattr(api, "_client")
