import importlib
import sys
import types

import pytest
from haiway import Meta


@pytest.mark.asyncio
async def test_fetch_preserves_object_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
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
    AWSS3Mixin = aws_s3.AWSS3Mixin

    class DummyS3(AWSS3Mixin):
        def __init__(
            self,
            region_name: str | None = None,
            access_key_id: str | None = None,
            secret_access_key: str | None = None,
        ) -> None:
            super().__init__(
                region_name=region_name,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
            )

        async def _fetch(self, bucket: str, name: str) -> bytes:
            assert bucket == "bucket"
            assert name == "path/to/object"
            return b"payload"

        async def _get_object_info(
            self,
            bucket: str,
            name: str,
        ) -> tuple[str | None, Meta]:
            assert bucket == "bucket"
            assert name == "path/to/object"
            return "text/plain", Meta.of({"foo": "bar", "name": "example"})

    content = await DummyS3().fetch("s3://bucket/path/to/object")

    assert content.meta["foo"] == "bar"
    assert content.meta["name"] == "example"
    assert content.mime_type == "text/plain"
    assert content.to_bytes() == b"payload"
