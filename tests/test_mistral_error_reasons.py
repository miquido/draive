import json
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest
from haiway import ctx
from mistralai.client.errors import SDKError

from draive.mistral.completions import MistralCompletions
from draive.mistral.config import MistralChatConfig, MistralEmbeddingConfig
from draive.mistral.embedding import MistralEmbedding
from draive.mistral.moderation import MistralContentModeration
from draive.mistral.utils import mistral_error_reason
from draive.models import ModelOutputFailed, ModelRateLimit, ModelTools
from draive.multimodal import TextContent

MARKER = "SECRET-MARKER-4f1c"


def _error(
    status: int,
    *,
    message: str = "Invalid model",
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> SDKError:
    payload: str = (
        body
        if body is not None
        else json.dumps(
            {
                "message": message,
                "echo": MARKER,
                "detail": [{"loc": ["body"], "input": MARKER}],
            }
        )
    )
    return SDKError(
        "API error occurred",
        httpx.Response(
            status,
            headers={"content-type": "application/json", **(headers or {})},
            text=payload,
        ),
    )


def _assert_safe(exc: ModelOutputFailed, error: SDKError, *, runner: Any) -> None:
    assert MARKER not in exc.reason
    assert "Body" not in exc.reason
    assert MARKER not in str(exc)
    # the embeddings failure is translated within a task group, which drops its cause
    if runner is not _embedding_failure:
        assert exc.__cause__ is error


async def _moderation_failure(error: SDKError) -> None:
    async def moderate_async(**_: Any) -> Any:
        raise error

    model: Any = object.__new__(MistralContentModeration)
    model._client = SimpleNamespace(classifiers=SimpleNamespace(moderate_async=moderate_async))
    async with ctx.scope("test"):
        await model.content_verification(TextContent.of("test"))


async def _embedding_failure(error: SDKError) -> None:
    async def create_async(**_: Any) -> Any:
        raise error

    model: Any = object.__new__(MistralEmbedding)
    model._client = SimpleNamespace(embeddings=SimpleNamespace(create_async=create_async))
    async with ctx.scope("test"):
        await model.create_texts_embedding(["a"], config=MistralEmbeddingConfig(model="embed"))


def _completion_runner() -> Any:
    async def run(error: SDKError) -> None:
        async def stream_async(**_: Any) -> Any:
            raise error

        model: Any = object.__new__(MistralCompletions)
        model._client = SimpleNamespace(chat=SimpleNamespace(stream_async=stream_async))
        async with ctx.scope("test"):
            async for _ in model.completion(
                instructions="",
                tools=ModelTools.none,
                context=(),
                output="text",
                config=MistralChatConfig(model="m"),
            ):
                pass

    return run


_RUNNERS: list[Any] = [_completion_runner(), _moderation_failure, _embedding_failure]
_IDS = ["completion", "moderation", "embedding"]


@pytest.mark.asyncio
@pytest.mark.parametrize("runner", _RUNNERS, ids=_IDS)
@pytest.mark.parametrize("status", [400, 422, 500])
async def test_reason_has_status_and_message_but_not_body(runner: Any, status: int) -> None:
    error = _error(status)
    with pytest.raises(ModelOutputFailed) as caught:
        await runner(error)

    assert f"HTTP {status}" in caught.value.reason
    assert "Invalid model" in caught.value.reason
    _assert_safe(caught.value, error, runner=runner)


@pytest.mark.asyncio
@pytest.mark.parametrize("runner", _RUNNERS, ids=_IDS)
@pytest.mark.parametrize("body", ["<html>" + MARKER * 5000 + "</html>", ""])
async def test_non_json_body_gives_only_status(runner: Any, body: str) -> None:
    error = _error(502, body=body)
    with pytest.raises(ModelOutputFailed) as caught:
        await runner(error)

    assert caught.value.reason == "HTTP 502"
    _assert_safe(caught.value, error, runner=runner)


@pytest.mark.asyncio
@pytest.mark.parametrize("runner", _RUNNERS, ids=_IDS)
async def test_long_message_is_truncated(runner: Any) -> None:
    error = _error(400, message="x" * 1000)
    with pytest.raises(ModelOutputFailed) as caught:
        await runner(error)

    assert "x" * 300 not in caught.value.reason.replace("x" * 299 + "…", "")
    assert len(caught.value.reason) < 320
    _assert_safe(caught.value, error, runner=runner)


@pytest.mark.asyncio
@pytest.mark.parametrize("runner", _RUNNERS, ids=_IDS)
async def test_rate_limit_is_still_mapped_and_keeps_retry_after(runner: Any) -> None:
    error = _error(429, headers={"Retry-After": "7"})
    with pytest.raises(ModelRateLimit) as caught:
        await runner(error)

    assert caught.value.retry_after == 7
    assert MARKER not in str(caught.value)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (json.dumps({"message": "bad"}), "HTTP 400 - bad"),
        (json.dumps({"message": "  a \n\t b  "}), "HTTP 400 - a b"),
        (json.dumps({"error": "oops"}), "HTTP 400 - oops"),
        (json.dumps({"detail": "nope"}), "HTTP 400 - nope"),
        (json.dumps({"message": {"x": MARKER}}), "HTTP 400"),
        (json.dumps({"message": 5, "detail": [MARKER]}), "HTTP 400"),
        (json.dumps({"detail": [{"input": MARKER}]}), "HTTP 400"),
        (json.dumps([MARKER]), "HTTP 400"),
        (json.dumps("text"), "HTTP 400"),
        (json.dumps({"message": ""}), "HTTP 400"),
        (json.dumps({"message": "zażółć gęślą jaźń"}), "HTTP 400 - zażółć gęślą jaźń"),
        ("not json " + MARKER, "HTTP 400"),
        ("", "HTTP 400"),
    ],
)
def test_mistral_error_reason_edge_cases(body: str, expected: str) -> None:
    assert mistral_error_reason(_error(400, body=body)) == expected


def test_mistral_error_reason_truncates_with_ellipsis() -> None:
    reason = mistral_error_reason(_error(400, message="y" * 500))

    assert reason == "HTTP 400 - " + "y" * 299 + "…"
