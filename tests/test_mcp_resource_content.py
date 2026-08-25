import json
from base64 import b64decode, b64encode

import pytest
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ReadResourceResult,
    TextResourceContents,
)

from draive.mcp.client import MCPClient, MCPClients
from draive.mcp.server import (  # pyright: ignore[reportPrivateUsage]
    _convert_multimodal_content,
    _resource_content,
)
from draive.multimodal import ArtifactContent, MultimodalContent
from draive.resources import Resource, ResourceContent

# bytes whose standard base64 uses `+` and `/`, spelled `-` and `_` by the urlsafe alphabet -
# the MCP wire format is standard base64, mixing the two silently corrupts the payload
_ALPHABET_SENSITIVE = bytes([0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF])


def _client() -> MCPClient:
    return MCPClient.stdio(command="true")


def test_text_resource_contents_are_encoded_not_decoded() -> None:
    # `text` holds the already decoded content while `ResourceContent.data` holds base64
    content = _client()._convert_resource_content(  # pyright: ignore[reportPrivateUsage]
        TextResourceContents(
            uri="file:///notes.txt",
            text="plain text, not base64",
            mime_type="text/plain",
        )
    )

    assert content.to_bytes() == b"plain text, not base64"
    assert content.mime_type == "text/plain"


def test_blob_resource_contents_keep_the_wire_payload() -> None:
    content = _client()._convert_resource_content(  # pyright: ignore[reportPrivateUsage]
        BlobResourceContents(
            uri="file:///payload.bin",
            blob=b64encode(_ALPHABET_SENSITIVE).decode(),
            mime_type="application/octet-stream",
        )
    )

    assert content.to_bytes() == _ALPHABET_SENSITIVE


def test_text_resource_round_trips_through_the_server() -> None:
    content = ResourceContent.of(b"plain text, not base64", mime_type="text/plain")

    [contents] = _resource_content(
        Resource.of(content, uri="file:///notes.txt"),
        uri="file:///notes.txt",
    )
    assert isinstance(contents, TextResourceContents)
    # the server hands over decoded text, the client encodes it back
    assert contents.text == "plain text, not base64"

    restored = _client()._convert_resource_content(contents)  # pyright: ignore[reportPrivateUsage]
    assert restored.data == content.data


def test_blob_resource_round_trips_through_the_server() -> None:
    content = ResourceContent.of(_ALPHABET_SENSITIVE, mime_type="application/octet-stream")

    [contents] = _resource_content(
        Resource.of(content, uri="file:///payload.bin"),
        uri="file:///payload.bin",
    )
    assert isinstance(contents, BlobResourceContents)
    # a standard base64 blob is what every other MCP peer expects to decode
    assert contents.blob == b64encode(_ALPHABET_SENSITIVE).decode()

    restored = _client()._convert_resource_content(contents)  # pyright: ignore[reportPrivateUsage]
    assert restored.to_bytes() == _ALPHABET_SENSITIVE


def test_artifact_content_is_embedded_as_standard_base64() -> None:
    # `?` lands on a base64 character the two alphabets disagree about
    artifact = {"value": "?"}

    [converted] = _convert_multimodal_content(
        MultimodalContent.of(ArtifactContent.of(artifact, category="json"))
    )

    assert isinstance(converted, EmbeddedResource)
    blob = converted.resource
    assert isinstance(blob, BlobResourceContents)
    # strict decoding rejects the urlsafe alphabet outright
    assert json.loads(b64decode(blob.blob, validate=True)) == artifact
    assert blob.uri == f"data:application/json;base64,{blob.blob}"


def test_blob_resource_content_is_embedded_as_standard_base64() -> None:
    [converted] = _convert_multimodal_content(
        MultimodalContent.of(
            ResourceContent.of(_ALPHABET_SENSITIVE, mime_type="application/octet-stream")
        )
    )

    assert isinstance(converted, EmbeddedResource)
    blob = converted.resource
    assert isinstance(blob, BlobResourceContents)
    assert b64decode(blob.blob, validate=True) == _ALPHABET_SENSITIVE


def test_text_resource_content_is_embedded_as_plain_text() -> None:
    [converted] = _convert_multimodal_content(
        MultimodalContent.of(ResourceContent.of(b"plain text", mime_type="text/plain"))
    )

    assert converted.type == "text"
    assert converted.text == "plain text"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "abc",
        # a length which is not a multiple of 4 breaks any base64 decoding attempt
        "Hello, world!",
        # a valid base64 string would be silently corrupted by decoding it
        "abcd",
        "zażółć gęślą jaźń",
    ],
)
def test_text_resource_round_trip_preserves_arbitrary_text(text: str) -> None:
    content = ResourceContent.of(text.encode(), mime_type="text/plain")

    [contents] = _resource_content(
        Resource.of(content, uri="file:///notes.txt"),
        uri="file:///notes.txt",
    )
    assert isinstance(contents, TextResourceContents)
    assert contents.text == text

    restored = _client()._convert_resource_content(contents)  # pyright: ignore[reportPrivateUsage]
    assert restored.data == content.data
    assert restored.to_bytes() == text.encode()


class _MultiContentSession:
    async def read_resource(self, uri: str) -> ReadResourceResult:
        _ = uri
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri="file:///a.txt",
                    text="a",
                    mime_type="text/plain",
                ),
                TextResourceContents(
                    uri="file:///b.txt",
                    text="b",
                    mime_type="text/plain",
                ),
            ]
        )


@pytest.mark.asyncio
async def test_multi_content_read_references_are_resolvable_back() -> None:
    client = MCPClient.stdio("source", command="true")
    client._session = _MultiContentSession()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    references = await client.resource_fetch("mcp://source/multi")
    assert isinstance(references, list)
    assert [reference.uri for reference in references] == [
        "file://source/a.txt",
        "file://source/b.txt",
    ]

    clients = MCPClients.__new__(MCPClients)
    clients._clients = {"source": client}  # pyright: ignore[reportPrivateUsage]
    # without the identifier those references could not be routed back to their client
    assert [
        clients._client_identifier_for_uri(reference.uri)  # pyright: ignore[reportPrivateUsage]
        for reference in references
    ] == ["source", "source"]
