from collections.abc import Sequence
from typing import Any, Literal, overload

from haiway import META_EMPTY, Meta, State, statemethod

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
    @overload
    @classmethod
    async def fetch_list(
        cls,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]: ...

    @overload
    async def fetch_list(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]: ...

    @statemethod
    async def fetch_list(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]:
        return await self.list_fetching(**extra)

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
    ) -> Resource | None: ...

    @overload
    async def fetch(
        self,
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
    async def fetch(
        self,
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

    @overload
    async def fetch(
        self,
        resource: ResourceDeclaration | str,
        /,
        *,
        required: Literal[True],
    ) -> Resource: ...

    @statemethod
    async def fetch(
        self,
        resource: ResourceDeclaration | str,
        /,
        *,
        default: Resource | None = None,
        required: bool = True,
    ) -> Resource | None:
        uri: str = resource if isinstance(resource, str) else resource.uri

        if fetched := await self.fetching(uri):
            return fetched

        elif required and default is None:
            raise ResourceMissing(f"Missing resource: '{uri}'")

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
    ) -> None:
        return await self.uploading(resource, **extra)

    list_fetching: ResourceListFetching
    fetching: ResourceFetching
    uploading: ResourceUploading
    meta: Meta = META_EMPTY
