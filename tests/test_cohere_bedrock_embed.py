import json
import sys
import time
import types
from asyncio import gather
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from draive.cohere.bedrock import CohereBedrock


class _FakeBody:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _FakeBedrockRuntime:
    def __init__(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self._headers = headers or {}
        self.requests: list[dict[str, Any]] = []
        self.closed: bool = False

    def invoke_model(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {
            "body": _FakeBody(self._payload),
            "ResponseMetadata": {"HTTPHeaders": self._headers},
        }

    def close(self) -> None:
        self.closed = True


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeBedrockRuntime,
) -> None:
    class _FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def client(self, service_name: str) -> _FakeBedrockRuntime:
            assert service_name == "bedrock-runtime"
            return client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=_FakeSession))


@pytest.fixture
def bedrock_runtime(monkeypatch: pytest.MonkeyPatch) -> _FakeBedrockRuntime:
    client = _FakeBedrockRuntime(
        payload={
            "id": "embed-1",
            "response_type": "embeddings_by_type",
            "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
            "texts": ["a", "b"],
        },
        headers={"x-amzn-bedrock-input-token-count": "7"},
    )

    _install_fake_boto3(monkeypatch, client)

    return client


@asynccontextmanager
async def _cohere_bedrock(
    aws_region: str | None = None,
) -> AsyncGenerator[CohereBedrock]:
    # the client is prepared when entering the context, exactly as CohereAPI does it
    client = CohereBedrock(aws_region=aws_region)
    async with client:
        yield client


@pytest.mark.asyncio
async def test_text_embed_sends_cohere_body_without_model(
    bedrock_runtime: _FakeBedrockRuntime,
) -> None:
    async with _cohere_bedrock(aws_region="us-east-1") as client:
        response = await client.embed(
            model="cohere.embed-english-v3",
            texts=["a", "b"],
            embedding_types=["float"],
            input_type="search_document",
        )

    assert len(bedrock_runtime.requests) == 1
    request = bedrock_runtime.requests[0]
    # the model travels as a Bedrock argument, never inside the body
    assert request["modelId"] == "cohere.embed-english-v3"
    assert request["contentType"] == "application/json"

    body = json.loads(request["body"])
    assert "model" not in body
    # this is the exact body the cohere SDK puts on the wire for the same call
    assert body == {
        "input_type": "search_document",
        "texts": ["a", "b"],
        "embedding_types": ["float"],
    }

    assert response.embeddings.float_ == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_image_embed_sends_images_body(bedrock_runtime: _FakeBedrockRuntime) -> None:
    async with _cohere_bedrock() as client:
        await client.embed(
            model="cohere.embed-english-v3",
            images=["data:image/jpeg;base64,QQ=="],
            embedding_types=["float"],
            input_type="image",
        )

    body = json.loads(bedrock_runtime.requests[0]["body"])
    assert body == {
        "input_type": "image",
        "images": ["data:image/jpeg;base64,QQ=="],
        "embedding_types": ["float"],
    }


@pytest.mark.asyncio
async def test_optional_arguments_are_omitted_when_unset(
    bedrock_runtime: _FakeBedrockRuntime,
) -> None:
    async with _cohere_bedrock() as client:
        await client.embed(
            model="cohere.embed-v4:0",
            texts=["a"],
            input_type="search_query",
        )

    body = json.loads(bedrock_runtime.requests[0]["body"])
    assert body == {"input_type": "search_query", "texts": ["a"]}


@pytest.mark.asyncio
async def test_optional_arguments_are_forwarded_when_set(
    bedrock_runtime: _FakeBedrockRuntime,
) -> None:
    async with _cohere_bedrock() as client:
        await client.embed(
            model="cohere.embed-v4:0",
            texts=["a"],
            input_type="search_query",
            embedding_types=["float"],
            output_dimension=256,
            truncate="END",
        )

    body = json.loads(bedrock_runtime.requests[0]["body"])
    assert body["output_dimension"] == 256
    assert body["truncate"] == "END"


@pytest.mark.asyncio
async def test_token_counts_are_read_from_bedrock_headers(
    bedrock_runtime: _FakeBedrockRuntime,
) -> None:
    async with _cohere_bedrock() as client:
        response = await client.embed(
            model="cohere.embed-english-v3",
            texts=["a"],
            input_type="search_document",
        )

    assert len(bedrock_runtime.requests) == 1
    assert response.meta is not None
    assert response.meta.tokens is not None
    assert response.meta.tokens.input_tokens == 7


@pytest.mark.asyncio
async def test_response_without_identifier_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # `id` is required by the cohere response model, Bedrock does not always send it
    _install_fake_boto3(
        monkeypatch,
        _FakeBedrockRuntime(payload={"embeddings": {"float": [[1.0]]}}),
    )

    async with _cohere_bedrock() as client:
        response = await client.embed(
            model="cohere.embed-english-v3",
            texts=["a"],
            input_type="search_document",
        )

    assert response.embeddings.float_ == [[1.0]]


@pytest.mark.asyncio
async def test_concurrent_embeds_reuse_a_single_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # embedding gathers a request per batch - none of them may prepare another client,
    # every client beyond the first would leak, only the last one being closed
    created: list[_FakeBedrockRuntime] = []

    class _SlowSession:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def client(self, service_name: str) -> _FakeBedrockRuntime:
            assert service_name == "bedrock-runtime"
            # boto3 client preparation is slow enough to interleave with other requests
            time.sleep(0.05)
            client = _FakeBedrockRuntime(payload={"embeddings": {"float": [[1.0]]}})
            created.append(client)
            return client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=_SlowSession))

    async with _cohere_bedrock() as client:
        await gather(
            *(
                client.embed(
                    model="cohere.embed-v4:0",
                    texts=["a"],
                    input_type="search_document",
                )
                for _ in range(5)
            )
        )

    assert len(created) == 1
    assert created[0].closed
    assert len(created[0].requests) == 5
