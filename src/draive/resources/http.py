from collections.abc import Sequence
from typing import Any, Final

from haiway import HTTPClient, HTTPResponse, Meta

from draive.resources.types import ResourceContent, ResourceCorrupted, ResourceReference

__all__ = (
    "http_resource_deleting",
    "http_resource_fetching",
    "http_resource_list_fetching",
    "http_resource_uploading",
)


HTTP_OK: Final[int] = 200


async def http_resource_list_fetching(
    **extra: Any,
) -> Sequence[ResourceReference]:
    # no listing available
    raise NotImplementedError("Resource listing is not implemented")


async def http_resource_fetching(
    uri: str,
    **extra: Any,
) -> ResourceContent | None:
    if not uri.startswith("http"):
        return None

    response: HTTPResponse = await HTTPClient.get(
        url=uri,
        **extra,
    )

    if response.status_code != HTTP_OK:
        raise ResourceCorrupted(uri=uri)

    return ResourceContent.of(
        await response.body(),
        mime_type=response.headers.get(
            "Content-Type",
            "application/octet-stream",
        ),
    )


async def http_resource_uploading(
    uri: str,
    content: ResourceContent,
    **extra: Any,
) -> Meta:
    # no upload available
    raise NotImplementedError("Resource uploading is not implemented")


async def http_resource_deleting(
    uri: str,
    **extra: Any,
) -> None:
    # no delete available
    raise NotImplementedError("Resource deleting is not implemented")
