from collections.abc import Sequence
from typing import Any, Literal, overload

from haiway import State, ctx

from draive.resources.types import (
    Resource,
    ResourceDeclaration,
    ResourceFetching,
    ResourceListFetching,
    ResourceMissing,
)

__all__ = [
    "Resources",
]


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
        **extra: Any,
    ) -> Resource | None: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
        *,
        default: Resource,
        **extra: Any,
    ) -> Resource: ...

    @overload
    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
        *,
        required: Literal[True],
        **extra: Any,
    ) -> Resource: ...

    @classmethod
    async def fetch(
        cls,
        resource: ResourceDeclaration | str,
        /,
        *,
        default: Resource | None = None,
        required: bool = True,
        **extra: Any,
    ) -> Resource | None:
        uri: str = resource if isinstance(resource, str) else resource.uri

        if fetched := await ctx.state(cls).fetching(
            uri,
            **extra,
        ):
            return fetched

        elif required and default is None:
            raise ResourceMissing(f"Missing resource: '{uri}'")

        else:
            return default

    list_fetching: ResourceListFetching
    fetching: ResourceFetching
