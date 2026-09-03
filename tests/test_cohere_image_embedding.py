import json
import sys
import types
from base64 import b64decode
from typing import Any

import pytest

from draive.cohere.config import CohereImageEmbeddingConfig
from draive.cohere.embedding import CohereEmbedding

# bytes whose standard base64 uses `+` and `/`, spelled `-` and `_` by the urlsafe alphabet -
# a data URI carries standard base64, mixing the two silently corrupts the image
_ALPHABET_SENSITIVE = bytes([0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF])


class _FakeBody:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _FakeBedrockRuntime:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def invoke_model(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {
            "body": _FakeBody({"embeddings": {"float": [[0.1]]}}),
            "ResponseMetadata": {"HTTPHeaders": {}},
        }

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_image_data_uri_uses_the_standard_base64_alphabet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeBedrockRuntime()

    class _FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def client(self, service_name: str) -> _FakeBedrockRuntime:
            assert service_name == "bedrock-runtime"
            return runtime

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(Session=_FakeSession))

    embedding = CohereEmbedding("bedrock", aws_region="us-east-1")
    embedding._client = embedding._prepare_client()  # pyright: ignore[reportPrivateUsage]
    await embedding._client.__aenter__()  # pyright: ignore[reportPrivateUsage]
    try:
        await embedding.create_images_embedding(
            [_ALPHABET_SENSITIVE],
            config=CohereImageEmbeddingConfig(model="cohere.embed-v4:0"),
        )

    finally:
        await embedding._client.__aexit__(None, None, None)  # pyright: ignore[reportPrivateUsage]

    body = json.loads(runtime.requests[0]["body"])
    [data_uri] = body["images"]
    prefix, _, encoded = data_uri.partition(",")
    assert prefix == "data:image/jpeg;base64"
    # decoding the way any recipient of a data URI does has to yield the original bytes
    assert b64decode(encoded) == _ALPHABET_SENSITIVE
