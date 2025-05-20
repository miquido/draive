from collections.abc import Sequence
from typing import Any, Literal, overload

from haiway import State, ctx

from draive.commons import META_EMPTY, Meta
from draive.resources.types import (
    Resource,
    ResourceDeclaration,
    ResourceFetching,
    ResourceListFetching,
    ResourceMissing,
    ResourceUploading,
)

__all__ = ("Resources",)


class Resources(State):
    @classmethod
    async def fetch_list(
        cls,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]:
        return await ctx.state(cls).list_fetching(**extra)

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
    ) -> Resource | None: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
        *,
        default: Resource,
    ) -> Resource: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
        *,
        required: Literal[True],
    ) -> Resource: ...

    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
        *,
        default: Resource | None = None,
        required: bool = True,
    ) -> Resource | None:
        uri: str = resource if isinstance(resource, str) else resource.uri

        if fetched := await ctx.state(cls).fetching(uri):
            return fetched

        elif required and default is None:
            raise ResourceMissing(f"Missing resource: '{uri}'")

        else:
            return default

    @classmethod
    async def upload(
        cls,
        resource: Resource,
        /,
        **extra: Any,
    ) -> None:
        return await ctx.state(cls).uploading(resource, **extra)

    list_fetching: ResourceListFetching
    fetching: ResourceFetching
    uploading: ResourceUploading
    meta: Meta = META_EMPTY
