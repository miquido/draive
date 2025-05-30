from collections.abc import Sequence
from typing import Any, Protocol, Self, overload, runtime_checkable

from haiway import State

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.multimodal import (
    MediaData,
    MediaReference,
    MultimodalContent,
    MultimodalContentElement,
    TextContent,
)
from draive.parameters import DataModel

__all__ = (
    "Resource",
    "ResourceContent",
    "ResourceDeclaration",
    "ResourceException",
    "ResourceFetching",
    "ResourceListFetching",
    "ResourceMissing",
    "ResourceTemplateDeclaration",
    "ResourceUploading",
)


class ResourceException(Exception):
    pass


class ResourceMissing(ResourceException):
    pass


class ResourceDeclaration(DataModel):
    uri: str
    mime_type: str | None
    name: str
    description: str | None = None
    meta: Meta = META_EMPTY


class ResourceTemplateDeclaration(DataModel):
    uri_template: str
    mime_type: str | None
    name: str
    description: str | None = None
    meta: Meta = META_EMPTY


class ResourceContent(State):
    @overload
    @classmethod
    def of(
        cls,
        content: bytes,
        /,
        *,
        mime_type: str,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        content: MultimodalContentElement | str,
        /,
        *,
        mime_type: str | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        content: MultimodalContentElement | str | bytes,
        /,
        *,
        mime_type: str | None = None,
    ) -> Self:
        match content:
            case str() as text:
                return cls(
                    mime_type=mime_type or "text/plain",
                    blob=text.encode(),
                )

            case bytes() as data:
                return cls(
                    mime_type=mime_type or "application/octet-stream",
                    blob=data,
                )

            case TextContent() as text_content:
                return cls(
                    mime_type=mime_type or "text/plain",
                    blob=text_content.text.encode(),
                )

            case MediaData() as media_data:
                return cls(
                    mime_type=mime_type or media_data.media,
                    blob=media_data.data,
                )

            case MediaReference():
                raise ValueError("Resource can't use a media reference")

            case other:
                return cls(
                    mime_type=mime_type or "application/json",
                    blob=other.to_json().encode(),
                )

        return cls()

    mime_type: str
    blob: bytes

    def to_multimodal(self) -> MultimodalContentElement:
        match self.mime_type:
            case "text/plain":
                return TextContent(text=self.blob.decode())

            case "application/json":
                return DataModel.from_json(self.blob.decode())

            case other:
                # try to match supported media
                return MediaData.of(
                    self.blob,
                    media=other,
                )


class Resource(State):
    @overload
    @classmethod
    def of(
        cls,
        content: bytes,
        /,
        *,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        content: MultimodalContentElement | str,
        /,
        *,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        content: MultimodalContentElement | str | bytes,
        /,
        *,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        resource_content: ResourceContent
        resource_meta: Meta
        match content:
            case str() as text:
                resource_content = ResourceContent(
                    mime_type=mime_type or "text/plain",
                    blob=text.encode(),
                )
                resource_meta = Meta.of(meta)

            case bytes() as data:
                resource_content = ResourceContent(
                    mime_type=mime_type or "application/octet-stream",
                    blob=data,
                )
                resource_meta = Meta.of(meta)

            case TextContent() as text_content:
                resource_content = ResourceContent(
                    mime_type=mime_type or "text/plain",
                    blob=text_content.text.encode(),
                )
                resource_meta = (
                    text_content.meta.merged_with(meta) if meta is not None else text_content.meta
                )

            case MediaData() as media_data:
                resource_content = ResourceContent(
                    mime_type=mime_type or media_data.media,
                    blob=media_data.data,
                )
                resource_meta = (
                    media_data.meta.merged_with(meta) if meta is not None else media_data.meta
                )

            case MediaReference():
                raise ValueError("Resource can't use a media reference")

            case other:
                resource_content = ResourceContent(
                    mime_type=mime_type or "application/json",
                    blob=other.to_json().encode(),
                )
                resource_meta = Meta.of(meta)

        return cls(
            uri=uri,
            name=name or uri,
            description=description,
            content=resource_content,
            meta=resource_meta,
        )

    uri: str
    name: str
    description: str | None
    content: Sequence[Self] | ResourceContent
    meta: Meta = META_EMPTY

    def to_multimodal(self) -> MultimodalContent:
        match self.content:
            case ResourceContent() as content:
                return MultimodalContent.of(content.to_multimodal())

            case content:
                return MultimodalContent.of(*[element.to_multimodal() for element in content])


@runtime_checkable
class ResourceFetching(Protocol):
    async def __call__(
        self,
        uri: str,
    ) -> Resource | None: ...


@runtime_checkable
class ResourceListFetching(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]: ...


@runtime_checkable
class ResourceUploading(Protocol):
    async def __call__(
        self,
        resource: Resource,
        **extra: Any,
    ) -> None: ...
