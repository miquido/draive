from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

from haiway import ctx

from draive.resources.state import ResourceRepository
from draive.resources.types import MissingResource, Resource, ResourceDeclaration

__all__ = [
    "fetch_resource",
    "fetch_resource_list",
]


async def fetch_resource_list(
    **extra: Any,
) -> Sequence[ResourceDeclaration]:
    return await ctx.state(ResourceRepository).list(**extra)


@overload
async def fetch_resource(
    reference: ResourceDeclaration | str,
    /,
    *,
    default: Resource | None = None,
    variables: Mapping[str, str] | None = None,
    **extra: Any,
) -> Resource | None: ...


@overload
async def fetch_resource(
    reference: ResourceDeclaration | str,
    /,
    *,
    default: Resource,
    **extra: Any,
) -> Resource: ...


@overload
async def fetch_resource(
    reference: ResourceDeclaration | str,
    /,
    *,
    default: Resource | None = None,
    required: Literal[True],
    **extra: Any,
) -> Resource: ...


async def fetch_resource(
    reference: ResourceDeclaration | str,
    /,
    *,
    default: Resource | None = None,
    required: bool = True,
    **extra: Any,
) -> Resource | None:
    uri: str = reference if isinstance(reference, str) else reference.uri

    match await ctx.state(ResourceRepository).fetch(
        uri,
        **extra,
    ):
        case None:
            match default:
                case None:
                    if required:
                        raise MissingResource(f"Missing resource: '{uri}'")

                    else:
                        return None

                case Resource() as resource:
                    return resource

        case resource:
            return resource
