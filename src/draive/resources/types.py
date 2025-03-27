from collections.abc import Sequence
from typing import Any, Protocol, Self, overload, runtime_checkable

from haiway import Default, State

from draive.commons import META_EMPTY, Meta
from draive.multimodal import (
    MediaContent,
    MultimodalContentElement,
    TextContent,
    # validated_media_type,
)
from draive.parameters import DataModel

__all__ = [
    "Resource",
    "ResourceContent",
    "ResourceDeclaration",
    "ResourceException",
    "ResourceFetching",
    "ResourceListFetching",
    "ResourceMissing",
]


class ResourceException(Exception):
    pass


class ResourceMissing(ResourceException):
    pass


class ResourceDeclaration(DataModel):
    uri: str
    mime_type: str | None
    name: str
    description: str | None = None
    meta: Meta = Default(META_EMPTY)


# TODO: add ResourceTemplateDeclaration for ResourceTemplates support
# https://modelcontextprotocol.io/docs/concepts/resources#resource-templates


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

            case MediaContent() as media_content:
                match media_content.source:
                    case bytes() as data:
                        return cls(
                            mime_type=mime_type or media_content.media,
                            blob=data,
                        )

                    case _:
                        raise ValueError("Resource can't use a content reference")

            case other:
                return cls(
                    mime_type=mime_type or "application/json",
                    blob=other.as_json().encode(),
                )

        return cls()

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
    meta: Meta = Default(META_EMPTY)


@runtime_checkable
class ResourceFetching(Protocol):
    async def __call__(
        self,
        uri: str,
        **extra: Any,
    ) -> Resource | None: ...


@runtime_checkable
class ResourceListFetching(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]: ...
