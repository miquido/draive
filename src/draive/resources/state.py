from haiway import State

from draive.resources.types import ResourceFetching, ResourceListing

__all__ = [
    "ResourceRepository",
]


class ResourceRepository(State):
    list: ResourceListing
    fetch: ResourceFetching
