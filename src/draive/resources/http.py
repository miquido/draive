from typing import Any, Final
from urllib.parse import urlparse

from haiway import HTTPClient, HTTPResponse, Meta, Paginated, Pagination

from draive.resources.types import (
    ResourceContent,
    ResourceCorrupted,
    ResourceInaccessible,
    ResourceReference,
    ResourceUnresolveable,
)

__all__ = (
    "http_resource_deleting",
    "http_resource_fetching",
    "http_resource_list_fetching",
    "http_resource_uploading",
)


HTTP_OK: Final[int] = 200
HTTP_CREATED: Final[int] = 201
HTTP_ACCEPTED: Final[int] = 202
HTTP_NO_CONTENT: Final[int] = 204
HTTP_UNAUTHORIZED: Final[int] = 401
HTTP_FORBIDDEN: Final[int] = 403
HTTP_NOT_FOUND: Final[int] = 404


async def http_resource_list_fetching(
    *,
    pagination: Pagination | None,
    **extra: Any,
) -> Paginated[ResourceReference]:
    # no listing available
    raise NotImplementedError("Resource listing is not implemented")


async def http_resource_fetching(
    uri: str,
    **extra: Any,
) -> ResourceContent | None:
    if urlparse(uri).scheme.lower() not in {"http", "https"}:
        raise ResourceUnresolveable(uri=uri)

    response: HTTPResponse = await HTTPClient.get(
        url=uri,
        **extra,
    )

    if response.status_code == HTTP_NOT_FOUND:
        return None

    if response.status_code == HTTP_UNAUTHORIZED:
        raise ResourceInaccessible(
            uri=uri,
            description="Unauthorized",
        )

    if response.status_code == HTTP_FORBIDDEN:
        raise ResourceInaccessible(
            uri=uri,
            description="Forbidden",
        )

    if response.status_code != HTTP_OK:
        raise ResourceCorrupted(uri=uri)

    return ResourceContent.of(
        await response.body(),
        mime_type=response.headers.get(
            "content-type",
            "application/octet-stream",
        ),
    )


async def http_resource_uploading(
    uri: str,
    content: ResourceContent,
    **extra: Any,
) -> Meta:
    if urlparse(uri).scheme.lower() not in {"http", "https"}:
        raise ResourceUnresolveable(uri=uri)

    response: HTTPResponse = await HTTPClient.put(
        url=uri,
        body=content.to_bytes(),
        headers={
            "content-type": content.mime_type,
            **(extra.pop("headers", None) or {}),
        },
        **extra,
    )

    if response.status_code == HTTP_NOT_FOUND:
        raise ResourceCorrupted(uri=uri)

    if response.status_code == HTTP_UNAUTHORIZED:
        raise ResourceInaccessible(
            uri=uri,
            description="Unauthorized",
        )

    if response.status_code == HTTP_FORBIDDEN:
        raise ResourceInaccessible(
            uri=uri,
            description="Forbidden",
        )

    if response.status_code not in {HTTP_OK, HTTP_CREATED, HTTP_ACCEPTED, HTTP_NO_CONTENT}:
        raise ResourceCorrupted(uri=uri)

    return Meta.of(
        {
            "status_code": response.status_code,
            **response.headers,
        }
    )


async def http_resource_deleting(
    uri: str,
    **extra: Any,
) -> None:
    if urlparse(uri).scheme.lower() not in {"http", "https"}:
        raise ResourceUnresolveable(uri=uri)

    response: HTTPResponse = await HTTPClient.request(
        "DELETE",
        url=uri,
        **extra,
    )

    if response.status_code == HTTP_UNAUTHORIZED:
        raise ResourceInaccessible(
            uri=uri,
            description="Unauthorized",
        )

    if response.status_code == HTTP_FORBIDDEN:
        raise ResourceInaccessible(
            uri=uri,
            description="Forbidden",
        )

    if response.status_code not in {HTTP_OK, HTTP_ACCEPTED, HTTP_NO_CONTENT, HTTP_NOT_FOUND}:
        raise ResourceCorrupted(uri=uri)
