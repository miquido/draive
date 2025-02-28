from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, runtime_checkable

from haiway import State

from draive.multimodal import (
    MediaContent,
    MultimodalContentElement,
    TextContent,
    # validated_media_type,
)
from draive.parameters import DataModel

__all__ = [
    "MissingResource",
    "Resource",
    "ResourceContent",
    "ResourceDeclaration",
    "ResourceFetching",
    "ResourceListing",
]


class MissingResource(Exception):
    pass


class ResourceDeclaration(DataModel):
    uri: str
    mime_type: str | None
    name: str
    description: str | None = None
    meta: Mapping[str, str | float | int | bool | None] | None


# TODO: add ResourceTemplateDeclaration for ResourceTemplates support
# https://modelcontextprotocol.io/docs/concepts/resources#resource-templates


class ResourceContent(State):
    mime_type: str
    blob: bytes

    def as_multimodal(self) -> MultimodalContentElement:
        match self.mime_type:
            case "text/plain":
                return TextContent(text=self.blob.decode())

            case "application/json":
                return DataModel.from_json(self.blob.decode())

            case other:
                # try to match supported media
                return MediaContent.data(
                    self.blob,
                    media=other,
                )


class Resource(State):
    uri: str
    name: str
    description: str | None
    content: Sequence[Self] | ResourceContent
    meta: Mapping[str, str | float | int | bool | None] | None


@runtime_checkable
class ResourceListing(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]: ...


@runtime_checkable
class ResourceFetching(Protocol):
    async def __call__(
        self,
        uri: str,
        **extra: Any,
    ) -> Resource | None: ...
