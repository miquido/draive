import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError
from haiway import Meta, Pagination, ctx

from draive.aws.s3 import AWSS3Mixin, _sanitize_metadata, _translate_client_error  # pyright: ignore[reportPrivateUsage]
from draive.aws.types import AWSAccessDenied, AWSError, AWSInvalidArgument, AWSResourceNotFound
from draive.resources import ResourceContent


def _client_error(code: str, status: int = 400) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "failure"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "operation",
    )


class _FakeS3Client:
    def __init__(self) -> None:
        self.failure: ClientError | None = None
        self.objects: dict[str, bytes] = {}
        self.put: list[dict[str, Any]] = []
        self.list_requests: list[dict[str, Any]] = []
        self.list_response: Mapping[str, Any] = {}
        self.head_response: Mapping[str, Any] = {"ContentType": "text/plain", "Metadata": {}}

    def download_fileobj(self, *, Bucket: str, Key: str, Fileobj: Any) -> None:
        if self.failure is not None:
            raise self.failure

        Fileobj.write(self.objects[f"{Bucket}/{Key}"])

    def head_object(self, **kwargs: Any) -> Mapping[str, Any]:
        if self.failure is not None:
            raise self.failure

        return self.head_response

    def put_object(self, **kwargs: Any) -> None:
        if self.failure is not None:
            raise self.failure

        self.put.append(kwargs)

    def list_objects_v2(self, **kwargs: Any) -> Mapping[str, Any]:
        if self.failure is not None:
            raise self.failure

        self.list_requests.append(kwargs)
        return self.list_response

    def list_buckets(self, **kwargs: Any) -> Mapping[str, Any]:
        if self.failure is not None:
            raise self.failure

        return {"Buckets": [{"Name": "alpha"}, {"Name": ""}, {"Name": "beta"}]}


def _mixin(client: _FakeS3Client) -> AWSS3Mixin:
    mixin = AWSS3Mixin(region_name="eu-west-1")
    mixin._s3_client = client  # pyright: ignore[reportPrivateUsage]
    return mixin


@pytest.mark.asyncio
async def test_fetch_returns_content_with_mime_type_and_metadata() -> None:
    client = _FakeS3Client()
    client.objects["bucket/dir/file.txt"] = b"hello"
    client.head_response = {"ContentType": "text/plain", "Metadata": {"author": "me"}}

    async with ctx.scope("test"):
        content = await _mixin(client).fetch("s3://bucket/dir/file.txt")

    assert isinstance(content, ResourceContent)
    assert content.to_bytes() == b"hello"
    assert content.mime_type == "text/plain"
    assert content.meta.get("author") == "me"


@pytest.mark.asyncio
async def test_fetch_defaults_to_octet_stream_without_content_type() -> None:
    client = _FakeS3Client()
    client.objects["bucket/blob"] = b"\x00\x01"
    client.head_response = {"Metadata": None}

    async with ctx.scope("test"):
        content = await _mixin(client).fetch("s3://bucket/blob")

    assert isinstance(content, ResourceContent)
    assert content.mime_type == "application/octet-stream"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("fetch", "download", "upload", "list"))
async def test_non_s3_uri_is_rejected(method: str, tmp_path: Path) -> None:
    mixin = _mixin(_FakeS3Client())

    async with ctx.scope("test"):
        with pytest.raises(ValueError, match="Unsupported"):
            match method:
                case "fetch":
                    await mixin.fetch("https://bucket/key")

                case "download":
                    await mixin.download("https://bucket/key", save_path=tmp_path / "out")

                case "upload":
                    await mixin.upload(
                        "https://bucket/key",
                        ResourceContent.of(b"a", mime_type="text/plain"),
                    )

                case _:
                    await mixin.list(uri="https://bucket/key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "status", "expected"),
    (
        ("NoSuchKey", 404, AWSResourceNotFound),
        ("NotFound", 404, AWSResourceNotFound),
        ("AccessDenied", 403, AWSAccessDenied),
        ("SignatureDoesNotMatch", 403, AWSAccessDenied),
        ("InternalError", 500, AWSError),
    ),
)
async def test_fetch_translates_errors(
    code: str,
    status: int,
    expected: type[Exception],
) -> None:
    client = _FakeS3Client()
    client.failure = _client_error(code, status)

    async with ctx.scope("test"):
        with pytest.raises(expected) as raised:
            await _mixin(client).fetch("s3://bucket/key")

    assert type(raised.value) is expected


@pytest.mark.asyncio
async def test_upload_sends_content_type_and_sanitized_metadata() -> None:
    client = _FakeS3Client()

    async with ctx.scope("test"):
        result = await _mixin(client).upload(
            "s3://bucket/dir/file.txt",
            ResourceContent.of(
                b"hello",
                mime_type="text/plain",
                meta=Meta.of({"note": "line\nbreak", "empty": ""}),
            ),
        )

    assert result == Meta.empty
    assert client.put == [
        {
            "Bucket": "bucket",
            "Key": "dir/file.txt",
            "Body": b"hello",
            "ContentType": "text/plain",
            "Metadata": {"note": "line break"},
        }
    ]


@pytest.mark.asyncio
async def test_download_writes_file(tmp_path: Path) -> None:
    client = _FakeS3Client()
    client.objects["bucket/key"] = b"payload"
    target = tmp_path / "out.bin"

    async with ctx.scope("test"):
        await _mixin(client).download("s3://bucket/key", save_path=target)

    assert target.read_bytes() == b"payload"


@pytest.mark.asyncio
async def test_download_translates_errors(tmp_path: Path) -> None:
    client = _FakeS3Client()
    client.failure = _client_error("NoSuchKey", 404)

    async with ctx.scope("test"):
        with pytest.raises(AWSResourceNotFound):
            await _mixin(client).download("s3://bucket/key", save_path=tmp_path / "out.bin")


@pytest.mark.asyncio
async def test_failed_download_does_not_leave_file_behind(tmp_path: Path) -> None:
    client = _FakeS3Client()
    client.failure = _client_error("NoSuchKey", 404)
    target = tmp_path / "out.bin"

    async with ctx.scope("test"):
        with pytest.raises(AWSResourceNotFound):
            await _mixin(client).download("s3://bucket/key", save_path=target)

    assert not target.exists()


@pytest.mark.asyncio
async def test_list_buckets_access_denied_is_translated() -> None:
    client = _FakeS3Client()
    client.failure = _client_error("AccessDenied", 403)

    async with ctx.scope("test"):
        with pytest.raises(AWSAccessDenied):
            await _mixin(client).list()


@pytest.mark.asyncio
async def test_list_buckets_skips_unnamed_and_applies_limit() -> None:
    async with ctx.scope("test"):
        references = await _mixin(_FakeS3Client()).list(limit=1)

    assert [reference.uri for reference in references] == ["s3://alpha"]


@pytest.mark.asyncio
async def test_list_paginated_buckets_token_walks_all_buckets() -> None:
    mixin = _mixin(_FakeS3Client())

    async with ctx.scope("test"):
        first = await mixin.list_paginated(pagination=Pagination.of(limit=1))
        assert first.pagination.token == "buckets:1"
        second = await mixin.list_paginated(pagination=first.pagination)
        assert second.pagination.token is None
        assert [item.uri for item in (*first.items, *second.items)] == [
            "s3://alpha",
            "s3://beta",
        ]


@pytest.mark.asyncio
async def test_list_paginated_rejects_foreign_bucket_token() -> None:
    async with ctx.scope("test"):
        with pytest.raises(ValueError, match="Invalid buckets pagination token"):
            await _mixin(_FakeS3Client()).list_paginated(
                pagination=Pagination.of(limit=1, token="other")
            )


@pytest.mark.asyncio
async def test_list_paginated_objects_forwards_continuation_token() -> None:
    client = _FakeS3Client()
    client.list_response = {
        "Contents": [
            {"Key": "dir/", "Size": 0},
            {"Key": "dir/a.txt", "Size": 3, "ETag": '"abc"'},
        ],
        "CommonPrefixes": [{"Prefix": "dir/sub/"}],
        "NextContinuationToken": "next",
    }

    async with ctx.scope("test"):
        page = await _mixin(client).list_paginated(
            uri="s3://bucket/dir/",
            pagination=Pagination.of(limit=5000, token="current"),
            recursive=False,
        )

    assert client.list_requests == [
        {
            "Bucket": "bucket",
            "Prefix": "dir/",
            "MaxKeys": 1000,
            "ContinuationToken": "current",
            "Delimiter": "/",
        }
    ]
    assert [item.uri for item in page.items] == [
        "s3://bucket/dir/a.txt",
        "s3://bucket/dir/sub/",
    ]
    assert page.pagination.token == "next"
    assert page.items[0].meta.get("etag") == "abc"


@pytest.mark.asyncio
async def test_listed_reference_with_special_characters_round_trips() -> None:
    client = _FakeS3Client()
    client.list_response = {"Contents": [{"Key": "dir/a#b?.txt"}]}
    client.objects["bucket/dir/a#b?.txt"] = b"data"

    async with ctx.scope("test"):
        mixin = _mixin(client)
        page = await mixin.list_paginated(uri="s3://bucket/dir/")
        content = await mixin.fetch(page.items[0].uri)

    assert isinstance(content, ResourceContent)
    assert content.to_bytes() == b"data"


def test_no_such_bucket_error_points_to_the_bucket() -> None:
    error = _translate_client_error(
        error=_client_error("NoSuchBucket", 404),
        bucket="bucket",
        key="key",
    )

    assert isinstance(error, AWSResourceNotFound)
    assert "s3://bucket/key" not in str(error)


def test_sanitize_metadata_truncates_by_bytes_and_drops_empty() -> None:
    sanitized = _sanitize_metadata(
        {
            "long": "z" * 2000,
            "blank": "   ",
            "tabs": "a\t\tb\x00c",
            "number": 3,
        }
    )

    assert set(sanitized) == {"long", "tabs", "number"}
    assert len(sanitized["long"].encode()) <= 1024
    assert sanitized["long"].endswith("...")
    assert sanitized["tabs"] == "a bc"
    assert sanitized["number"] == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    (
        "dir/with space.txt",
        "dir/zażółć.txt",
        "dir/100%.txt",
        "dir/a#b?c.txt",
        "dir/%2F%23.txt",
        "dir/a..b/../c.txt",
        "dir//double.txt",
        "dir/日本語 #1?.txt",
        "dir/a+b&c=d.txt",
    ),
)
async def test_reference_round_trips_through_fetch(key: str) -> None:
    client = _FakeS3Client()
    client.list_response = {"Contents": [{"Key": key}]}
    client.objects[f"bucket/{key}"] = b"data"

    async with ctx.scope("test"):
        mixin = _mixin(client)
        page = await mixin.list_paginated(uri="s3://bucket/dir/")
        content = await mixin.fetch(page.items[0].uri)

    assert isinstance(content, ResourceContent)
    assert content.to_bytes() == b"data"


@pytest.mark.asyncio
async def test_cancelled_download_does_not_leave_files_behind(tmp_path: Path) -> None:
    client = _FakeS3Client()
    client.failure = KeyboardInterrupt()

    async with ctx.scope("test"):
        with pytest.raises(KeyboardInterrupt):
            await _mixin(client).download("s3://bucket/key", save_path=tmp_path / "out.bin")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_list_with_zero_limit_returns_empty_without_calling_api() -> None:
    client = _FakeS3Client()
    client.list_response = {"Contents": [{"Key": "a"}]}
    mixin = _mixin(client)

    async with ctx.scope("test"):
        assert len(await mixin.list(uri="s3://bucket/prefix/", limit=0)) == 0
        assert len(await mixin.list(limit=0)) == 0

    assert client.list_requests == []


def test_sanitize_metadata_keeps_key_case_since_s3_normalizes_it() -> None:
    # BY DESIGN: S3 treats user metadata names case-insensitively and returns them lowercased
    assert _sanitize_metadata({"Foo-Bar": "v"}) == {"Foo-Bar": "v"}


@pytest.mark.parametrize(
    "meta",
    [{"zażółć": "ascii"}, {"plain": "gęślą jaźń"}],
)
def test_sanitize_metadata_rejects_non_ascii_naming_key_only(meta: Mapping[str, str]) -> None:
    with pytest.raises(AWSInvalidArgument) as exc_info:
        _sanitize_metadata(meta, uri="s3://bucket/key")

    message = str(exc_info.value)
    assert next(iter(meta)) in message
    assert "gęślą" not in message


@pytest.mark.asyncio
async def test_upload_with_non_ascii_metadata_raises_without_calling_client() -> None:
    client = _FakeS3Client()

    async with ctx.scope("test"):
        with pytest.raises(AWSInvalidArgument):
            await _mixin(client).upload(
                "s3://bucket/file.txt",
                ResourceContent.of(b"x", mime_type="text/plain", meta=Meta.of({"k": "zażółć"})),
            )

    assert client.put == []


@pytest.mark.asyncio
async def test_download_applies_umask_default_mode(tmp_path: Path) -> None:
    client = _FakeS3Client()
    client.objects["bucket/key"] = b"payload"
    target = tmp_path / "out.bin"
    previous = os.umask(0o022)
    try:
        async with ctx.scope("test"):
            await _mixin(client).download("s3://bucket/key", save_path=target)

    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


@pytest.mark.asyncio
async def test_download_preserves_mode_of_existing_target(tmp_path: Path) -> None:
    client = _FakeS3Client()
    client.objects["bucket/key"] = b"payload"
    target = tmp_path / "out.bin"
    target.write_bytes(b"old")
    target.chmod(0o640)

    async with ctx.scope("test"):
        await _mixin(client).download("s3://bucket/key", save_path=target)

    assert target.read_bytes() == b"payload"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert [path.name for path in tmp_path.iterdir()] == ["out.bin"]
