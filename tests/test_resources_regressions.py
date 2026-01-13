from typing import Any

import pytest

from draive.resources.http import (
    http_resource_deleting,
    http_resource_fetching,
    http_resource_uploading,
)
from draive.resources.state import ResourcesRepository
from draive.resources.types import (
    Resource,
    ResourceContent,
    ResourceCorrupted,
    ResourceInaccessible,
    ResourceReference,
    ResourceUnresolveable,
)


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_repository_fetch_wraps_content_in_resource() -> None:
    async def fake_fetching(uri: str, **extra: Any) -> ResourceContent:
        _ = extra
        assert uri == "memory://data"
        return ResourceContent.of(b"ok", mime_type="application/octet-stream")

    repository = ResourcesRepository(fetching=fake_fetching)

    fetched = await repository.fetch("memory://data")

    assert isinstance(fetched, Resource)
    assert isinstance(fetched.resource, ResourceContent)
    assert fetched.resource.to_bytes() == b"ok"


@pytest.mark.asyncio
async def test_repository_fetch_wraps_reference_collection() -> None:
    async def fake_fetching(uri: str, **extra: Any) -> tuple[ResourceReference, ...]:
        _ = (uri, extra)
        return (ResourceReference.of("memory://child", mime_type="text/plain"),)

    repository = ResourcesRepository(fetching=fake_fetching)

    fetched = await repository.fetch("memory://dir")

    assert isinstance(fetched, Resource)
    assert isinstance(fetched.resource, tuple)
    assert fetched.resource[0].uri == "memory://child"


@pytest.mark.asyncio
async def test_http_resource_fetching_rejects_non_http_scheme() -> None:
    with pytest.raises(ResourceUnresolveable):
        await http_resource_fetching("file:///tmp/test.txt")


@pytest.mark.asyncio
async def test_http_resource_fetching_returns_none_for_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(*, url: str, **kwargs: Any) -> _DummyResponse:
        _ = (url, kwargs)
        return _DummyResponse(status_code=404, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.get", fake_get)

    assert await http_resource_fetching("https://example.com/missing") is None


@pytest.mark.asyncio
async def test_http_resource_fetching_raises_for_non_ok_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(*, url: str, **kwargs: Any) -> _DummyResponse:
        _ = (url, kwargs)
        return _DummyResponse(status_code=201, body=b"created")

    monkeypatch.setattr("draive.resources.http.HTTPClient.get", fake_get)

    with pytest.raises(ResourceCorrupted):
        await http_resource_fetching("https://example.com/resource")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "description"),
    [
        (401, "Unauthorized"),
        (403, "Forbidden"),
    ],
)
async def test_http_resource_fetching_raises_inaccessible_for_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    description: str,
) -> None:
    async def fake_get(*, url: str, **kwargs: Any) -> _DummyResponse:
        _ = (url, kwargs)
        return _DummyResponse(status_code=status_code, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.get", fake_get)

    with pytest.raises(ResourceInaccessible) as exc_info:
        await http_resource_fetching("https://example.com/resource")

    assert exc_info.value.description == description


@pytest.mark.asyncio
async def test_http_resource_uploading_uses_put_and_returns_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_put(
        *,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> _DummyResponse:
        _ = kwargs
        assert url == "https://example.com/resource"
        assert body == b"payload"
        assert headers == {"content-type": "application/json", "x-test": "1"}
        return _DummyResponse(
            status_code=201,
            body=b"",
            headers={"etag": "abc"},
        )

    monkeypatch.setattr("draive.resources.http.HTTPClient.put", fake_put)

    meta = await http_resource_uploading(
        "https://example.com/resource",
        ResourceContent.of(b"payload", mime_type="application/json"),
        headers={"x-test": "1"},
    )

    assert meta["status_code"] == 201
    assert meta["etag"] == "abc"


@pytest.mark.asyncio
async def test_http_resource_uploading_raises_for_non_success_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_put(**kwargs: Any) -> _DummyResponse:
        _ = kwargs
        return _DummyResponse(status_code=500, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.put", fake_put)

    with pytest.raises(ResourceCorrupted):
        await http_resource_uploading(
            "https://example.com/resource",
            ResourceContent.of(b"payload", mime_type="application/octet-stream"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "description"),
    [
        (401, "Unauthorized"),
        (403, "Forbidden"),
    ],
)
async def test_http_resource_uploading_raises_inaccessible_for_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    description: str,
) -> None:
    async def fake_put(**kwargs: Any) -> _DummyResponse:
        _ = kwargs
        return _DummyResponse(status_code=status_code, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.put", fake_put)

    with pytest.raises(ResourceInaccessible) as exc_info:
        await http_resource_uploading(
            "https://example.com/resource",
            ResourceContent.of(b"payload", mime_type="application/octet-stream"),
        )

    assert exc_info.value.description == description


@pytest.mark.asyncio
async def test_http_resource_deleting_uses_delete_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_request(method: str, /, *, url: str, **kwargs: Any) -> _DummyResponse:
        _ = kwargs
        assert method == "DELETE"
        assert url == "https://example.com/resource"
        return _DummyResponse(status_code=204, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.request", fake_request)

    await http_resource_deleting("https://example.com/resource")


@pytest.mark.asyncio
async def test_http_resource_deleting_raises_for_non_success_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_request(method: str, /, *, url: str, **kwargs: Any) -> _DummyResponse:
        _ = (method, url, kwargs)
        return _DummyResponse(status_code=409, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.request", fake_request)

    with pytest.raises(ResourceCorrupted):
        await http_resource_deleting("https://example.com/resource")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "description"),
    [
        (401, "Unauthorized"),
        (403, "Forbidden"),
    ],
)
async def test_http_resource_deleting_raises_inaccessible_for_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    description: str,
) -> None:
    async def fake_request(method: str, /, *, url: str, **kwargs: Any) -> _DummyResponse:
        _ = (method, url, kwargs)
        return _DummyResponse(status_code=status_code, body=b"")

    monkeypatch.setattr("draive.resources.http.HTTPClient.request", fake_request)

    with pytest.raises(ResourceInaccessible) as exc_info:
        await http_resource_deleting("https://example.com/resource")

    assert exc_info.value.description == description
