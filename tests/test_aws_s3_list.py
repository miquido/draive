import importlib
import sys
import types
from collections.abc import Generator

import pytest


class _FakePaginator:
    def __init__(self, pages: list[dict[str, object]], client: "_FakeS3Client") -> None:
        self._pages = pages
        self._client = client

    def paginate(self, **request: object) -> Generator[dict[str, object]]:
        self._client.request = request
        yield from self._pages


class _FakeS3Client:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self._pages = pages
        self.request: dict[str, object] | None = None

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator(self._pages, self)

    def list_buckets(self, **extra: object) -> dict[str, object]:
        self.request = extra
        return {
            "Buckets": [
                {"Name": "alpha", "CreationDate": "2024-01-01T00:00:00Z"},
                {"Name": "beta"},
            ]
        }


@pytest.fixture
def patched_aws_modules(monkeypatch: pytest.MonkeyPatch) -> type:
    fake_boto = types.SimpleNamespace(
        Session=type(
            "FakeSession",
            (),
            {
                "__init__": lambda self, **kwargs: None,
                "client": lambda self, service_name: object(),
            },
        ),
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto)

    aws_api = importlib.import_module("draive.aws.api")
    importlib.reload(aws_api)

    aws_s3 = importlib.import_module("draive.aws.s3")
    importlib.reload(aws_s3)
    return aws_s3.AWSS3Mixin


@pytest.mark.asyncio
async def test_list_recursive_lists_objects(patched_aws_modules: type) -> None:
    pages = [
        {
            "Contents": [
                {
                    "Key": "prefix/a.txt",
                    "ETag": '"etag-a"',
                    "Size": 123,
                    "LastModified": "2024-01-01T00:00:00Z",
                    "StorageClass": "STANDARD",
                },
                {"Key": "prefix/sub/b.txt", "Size": 10},
            ],
        }
    ]

    class DummyS3(patched_aws_modules):
        def __init__(self) -> None:
            super().__init__()
            self._s3_client = _FakeS3Client(pages)

    client = DummyS3()
    references = await client.list(uri="s3://bucket/prefix/", recursive=True)

    assert [ref.uri for ref in references] == [
        "s3://bucket/prefix/a.txt",
        "s3://bucket/prefix/sub/b.txt",
    ]
    assert references[0].meta["type"] == "object"
    assert references[0].meta["mime_type"] == "text/plain"
    assert references[0].meta["etag"] == "etag-a"
    assert references[0].meta["size"] == 123
    assert references[0].meta["storage_class"] == "STANDARD"
    assert references[0].meta["last_modified"] == "2024-01-01T00:00:00Z"
    assert client._s3_client.request == {
        "Bucket": "bucket",
        "Prefix": "prefix/",
    }


@pytest.mark.asyncio
async def test_list_non_recursive_includes_prefixes(patched_aws_modules: type) -> None:
    pages = [
        {
            "Contents": [{"Key": "prefix/a.txt", "Size": 1}],
            "CommonPrefixes": [{"Prefix": "prefix/sub/"}],
        }
    ]

    class DummyS3(patched_aws_modules):
        def __init__(self) -> None:
            super().__init__()
            self._s3_client = _FakeS3Client(pages)

    client = DummyS3()
    references = await client.list(uri="s3://bucket/prefix/", recursive=False)

    assert [ref.uri for ref in references] == [
        "s3://bucket/prefix/a.txt",
        "s3://bucket/prefix/sub/",
    ]
    assert references[0].meta["type"] == "object"
    assert references[1].meta["type"] == "prefix"
    assert client._s3_client.request == {
        "Bucket": "bucket",
        "Prefix": "prefix/",
        "Delimiter": "/",
    }


@pytest.mark.asyncio
async def test_list_without_uri_lists_buckets(patched_aws_modules: type) -> None:
    class DummyS3(patched_aws_modules):
        def __init__(self) -> None:
            super().__init__()
            self._s3_client = _FakeS3Client([])

    client = DummyS3()
    references = await client.list()

    assert [ref.uri for ref in references] == ["s3://alpha", "s3://beta"]
    assert references[0].meta["type"] == "bucket"
    assert references[0].meta["creation_date"] == "2024-01-01T00:00:00Z"
    assert client._s3_client.request == {}
