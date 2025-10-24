from base64 import b64decode, b64encode
from collections.abc import Collection, Sequence
from typing import Any, Literal, Protocol, Self, final, overload, runtime_checkable

from haiway import META_EMPTY, Meta, MetaValues, State

from draive.parameters import DataModel

__all__ = (
    "MimeType",
    "Resource",
    "ResourceContent",
    "ResourceCorrupted",
    "ResourceDeleting",
    "ResourceFetching",
    "ResourceListFetching",
    "ResourceMissing",
    "ResourceReference",
    "ResourceReferenceTemplate",
    "ResourceUploading",
)


class ResourceMissing(Exception):
    """Raised when a resource cannot be found."""

    __slots__ = ("uri",)

    def __init__(
        self,
        *,
        uri: str,
    ) -> None:
        super().__init__(f"Missing resource - {uri}")
        self.uri: str = uri or ""


class ResourceCorrupted(Exception):
    """Raised when a resource cannot be resolved correctly."""

    __slots__ = ("uri",)

    def __init__(
        self,
        *,
        uri: str,
    ) -> None:
        super().__init__(f"Corrupted resource - {uri}")
        self.uri: str = uri or ""


MimeType = (
    Literal[
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/gif",
        "audio/aac",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        "audio/pcm16",
        "video/mp4",
        "video/mpeg",
        "video/ogg",
        "application/octet-stream",
        "application/json",
        "text/plain",
    ]
    | str
)


@final
class ResourceReference(DataModel):
    @classmethod
    def of(
        cls,
        /,
        uri: str,
        *,
        name: str | None = None,
        description: str | None = None,
        mime_type: MimeType | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            uri=uri,
            mime_type=mime_type,
            meta=Meta.of(meta).updated(
                name=name,
                description=description,
            ),
        )

    uri: str
    mime_type: MimeType | None = None
    meta: Meta = META_EMPTY


class ResourceReferenceTemplate(DataModel):
    @classmethod
    def of(
        cls,
        /,
        template_uri: str,
        *,
        name: str | None = None,
        description: str | None = None,
        mime_type: MimeType | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            template_uri=template_uri,
            mime_type=mime_type,
            meta=Meta.of(meta).updated(
                name=name,
                description=description,
            ),
        )

    template_uri: str
    mime_type: MimeType | None = None
    meta: Meta = META_EMPTY


@final
class ResourceContent(DataModel):
    @overload
    @classmethod
    def of(
        cls,
        content: bytes,
        /,
        *,
        mime_type: MimeType,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        content: str,
        /,
        *,
        mime_type: MimeType,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        content: str | bytes,
        /,
        *,
        mime_type: MimeType | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        if isinstance(content, str):
            # Treat plain strings as text content; encode to base64
            encoded_content: str = b64encode(content.encode("utf-8")).decode("utf-8")

            return cls(
                data=encoded_content,
                mime_type=mime_type if mime_type is not None else "text/plain",
                meta=Meta.of(meta),
            )

        else:
            assert isinstance(content, bytes)  # nosec: B101

            return cls(
                data=b64encode(content).decode("utf-8"),
                mime_type=mime_type if mime_type is not None else "application/octet-stream",
                meta=Meta.of(meta),
            )

    data: str  # base64 encoded
    mime_type: str
    meta: Meta = META_EMPTY

    def to_str(
        self,
        *,
        include_data: bool = False,
    ) -> str:
        kind: str
        if self.mime_type.startswith("image"):
            kind = "image"

        elif self.mime_type.startswith("audio"):
            kind = "audio"

        elif self.mime_type.startswith("video"):
            kind = "video"

        else:
            kind = ""

        if include_data:
            return f"![{kind}]({self.to_data_uri()})"

        else:
            return f"![{kind}](REDACTED)"

    def to_bytes(self) -> bytes:
        return b64decode(self.data)

    def to_data_uri(self) -> str:
        return f"data:{self.mime_type};base64,{self.data}"

    def __bool__(self) -> bool:
        return len(self.data) > 0


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
        mime_type: MimeType | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        content: str,
        /,
        *,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: MimeType | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        content: ResourceContent,
        /,
        *,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        content: Collection[ResourceReference] | ResourceContent | str | bytes,
        /,
        *,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: MimeType | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        resource_content: Collection[ResourceReference] | ResourceContent
        if isinstance(content, bytes):
            resource_content = ResourceContent.of(
                content,
                mime_type=mime_type if mime_type is not None else "application/octet-stream",
            )

        elif isinstance(content, str):
            resource_content = ResourceContent.of(
                content,  # assuming base64
                mime_type=mime_type if mime_type is not None else "application/octet-stream",
            )

        else:
            assert isinstance(content, Collection | ResourceContent)  # nosec: B101
            resource_content = content

        return cls(
            uri=uri,
            content=resource_content,
            meta=Meta.of(meta).updated(
                name=name,
                description=description,
            ),
        )

    uri: str
    content: Collection[ResourceReference] | ResourceContent
    meta: Meta = META_EMPTY

    @property
    def reference(self) -> ResourceReference:
        if isinstance(self.content, ResourceContent):
            return ResourceReference(
                uri=self.uri,
                mime_type=self.content.mime_type,
                meta=self.meta,
            )

        else:
            return ResourceReference(
                uri=self.uri,
                mime_type=None,
                meta=self.meta,
            )


@runtime_checkable
class ResourceFetching(Protocol):
    async def __call__(
        self,
        uri: str,
        **extra: Any,
    ) -> Collection[ResourceReference] | ResourceContent | None: ...


@runtime_checkable
class ResourceListFetching(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[ResourceReference]: ...


@runtime_checkable
class ResourceUploading(Protocol):
    async def __call__(
        self,
        uri: str,
        content: ResourceContent,
        **extra: Any,
    ) -> Meta: ...


@runtime_checkable
class ResourceDeleting(Protocol):
    async def __call__(
        self,
        uri: str,
        **extra: Any,
    ) -> None: ...
