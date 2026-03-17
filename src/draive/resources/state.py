from collections.abc import Collection
from typing import Any, Literal, final, overload

from haiway import Meta, Paginated, Pagination, State, statemethod

from draive.resources.http import (
    http_resource_deleting,
    http_resource_fetching,
    http_resource_list_fetching,
    http_resource_uploading,
)
from draive.resources.types import (
    Resource,
    ResourceContent,
    ResourceCorrupted,
    ResourceDeleting,
    ResourceFetching,
    ResourceListFetching,
    ResourceMissing,
    ResourceReference,
    ResourceUploading,
)

__all__ = ("ResourcesRepository",)


@final
class ResourcesRepository(State):
    @overload
    @classmethod
    async def fetch_list(
        cls,
        *,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[ResourceReference]: ...

    @overload
    async def fetch_list(
        self,
        *,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[ResourceReference]: ...

    @statemethod
    async def fetch_list(
        self,
        *,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        return await self.list_fetching(
            pagination=pagination,
            **extra,
        )

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceReference | str,
        /,
        **extra: Any,
    ) -> Resource | None: ...

    @overload
    async def fetch(
        self,
        resource: ResourceReference | str,
        /,
        **extra: Any,
    ) -> Resource | None: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceReference | str,
        /,
        *,
        default: Resource,
        **extra: Any,
    ) -> Resource: ...

    @overload
    async def fetch(
        self,
        resource: ResourceReference | str,
        /,
        *,
        default: Resource,
        **extra: Any,
    ) -> Resource: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceReference | str,
        /,
        *,
        required: Literal[True],
        **extra: Any,
    ) -> Resource: ...

    @overload
    async def fetch(
        self,
        resource: ResourceReference | str,
        /,
        *,
        required: Literal[True],
        **extra: Any,
    ) -> Resource: ...

    @statemethod
    async def fetch(
        self,
        resource: ResourceReference | str,
        /,
        *,
        default: Resource | None = None,
        required: bool = True,
        **extra: Any,
    ) -> Resource | None:
        uri: str
        if isinstance(resource, str):
            uri = resource

        else:
            uri = resource.uri

        fetched: Collection[ResourceReference] | ResourceContent | None = await self.fetching(
            uri,
            **extra,
        )

        if fetched is not None:
            return Resource(
                uri=uri,
                resource=fetched,
            )

        elif required and default is None:
            raise ResourceMissing(uri=uri)

        else:
            return default

    @overload
    @classmethod
    async def upload(
        cls,
        resource: Resource,
        /,
        **extra: Any,
    ) -> None: ...

    @overload
    async def upload(
        self,
        resource: Resource,
        /,
        **extra: Any,
    ) -> None: ...

    @statemethod
    async def upload(
        self,
        resource: Resource,
        /,
        **extra: Any,
    ) -> Meta:
        if isinstance(resource.resource, ResourceContent):
            return await self.uploading(
                resource.uri,
                resource.resource,
                **extra,
            )

        else:  # can't upload resource with references
            raise ResourceCorrupted(uri=resource.uri)

    @statemethod
    async def delete(
        self,
        resource: Resource,
        /,
        **extra: Any,
    ) -> None:
        return await self.deleting(
            resource.uri,
            **extra,
        )

    list_fetching: ResourceListFetching = http_resource_list_fetching
    fetching: ResourceFetching = http_resource_fetching
    uploading: ResourceUploading = http_resource_uploading
    deleting: ResourceDeleting = http_resource_deleting
    meta: Meta = Meta.empty
