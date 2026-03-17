from base64 import b64decode, b64encode
from collections.abc import Collection
from typing import Any, Literal, Protocol, Self, final, overload, runtime_checkable

from haiway import Meta, MetaValues, Paginated, Pagination, State

__all__ = (
    "MimeType",
    "Resource",
    "ResourceContent",
    "ResourceCorrupted",
    "ResourceDeleting",
    "ResourceException",
    "ResourceFetching",
    "ResourceInaccessible",
    "ResourceListFetching",
    "ResourceMissing",
    "ResourceReference",
    "ResourceReferenceTemplate",
    "ResourceUnresolveable",
    "ResourceUploading",
)


class ResourceException(Exception):
    __slots__ = ("uri",)

    def __init__(
        self,
        *args: object,
        uri: str,
    ) -> None:
        super().__init__(*args)
        self.uri: str = uri or ""


@final
class ResourceMissing(ResourceException):
    """Raised when a resource cannot be found."""

    def __init__(
        self,
        *,
        uri: str,
    ) -> None:
        super().__init__(
            f"Missing resource - {uri}",
            uri=uri,
        )


@final
class ResourceCorrupted(ResourceException):
    """Raised when a resource cannot be resolved correctly."""

    def __init__(
        self,
        *,
        uri: str,
    ) -> None:
        super().__init__(
            f"Corrupted resource - {uri}",
            uri=uri,
        )


@final
class ResourceInaccessible(ResourceException):
    """Raised when a resource cannot be accessed."""

    __slots__ = ("description",)

    def __init__(
        self,
        *,
        uri: str,
        description: str,
    ) -> None:
        super().__init__(
            f"Inaccessible resource ({description}) - {uri}",
            uri=uri,
        )
        self.description: str = description


@final
class ResourceUnresolveable(ResourceException):
    """Raised when a resource resolution method is not defined."""

    def __init__(
        self,
        *,
        uri: str,
    ) -> None:
        super().__init__(
            f"Unresolveable resource - {uri}",
            uri=uri,
        )


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
        "inode/directory",
        "application/octet-stream",
        "application/json",
        "text/plain",
    ]
    | str
)


@final
class ResourceReference(State, serializable=True):
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
            mime_type=mime_type if mime_type is not None else "application/octet-stream",
            meta=Meta.of(meta).updating(
                name=name,
                description=description,
            ),
        )

    uri: str
    mime_type: MimeType
    meta: Meta = Meta.empty


class ResourceReferenceTemplate(State, serializable=True):
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
            mime_type=mime_type if mime_type is not None else "application/octet-stream",
            meta=Meta.of(meta).updating(
                name=name,
                description=description,
            ),
        )

    template_uri: str
    mime_type: MimeType
    meta: Meta = Meta.empty


@final
class ResourceContent(State, serializable=True):
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
            # Treat strings as base64
            assert validate_base64(content)  # nosec: B101

            return cls(
                data=content,
                mime_type=mime_type if mime_type is not None else "application/octet-stream",
                meta=Meta.of(meta),
            )

        else:
            assert isinstance(content, bytes)  # nosec: B101

            return cls(
                data=b64encode(content).decode(),
                mime_type=mime_type if mime_type is not None else "application/octet-stream",
                meta=Meta.of(meta),
            )

    data: str  # base64 encoded
    mime_type: str
    meta: Meta = Meta.empty

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
            return f"![{kind}]()"

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
        content: str | bytes,
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
        content: Collection[ResourceReference] | ResourceContent,
        /,
        *,
        uri: str,
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
        if isinstance(content, str | bytes):
            resource_content = ResourceContent.of(
                content,
                mime_type=mime_type if mime_type is not None else "application/octet-stream",
                meta=Meta.of(meta).updating(
                    name=name,
                    description=description,
                ),
            )

        else:
            assert isinstance(content, Collection | ResourceContent)  # nosec: B101
            assert name is None and description is None and meta is None  # nosec: B101
            resource_content = content

        return cls(
            uri=uri,
            resource=resource_content,
        )

    uri: str
    resource: Collection[ResourceReference] | ResourceContent

    @property
    def reference(self) -> ResourceReference:
        if isinstance(self.resource, ResourceContent):
            return ResourceReference(
                uri=self.uri,
                mime_type=self.resource.mime_type,
            )

        else:
            return ResourceReference(
                uri=self.uri,
                mime_type="inode/directory",
            )


@runtime_checkable
class ResourceListFetching(Protocol):
    async def __call__(
        self,
        *,
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[ResourceReference]: ...


@runtime_checkable
class ResourceFetching(Protocol):
    async def __call__(
        self,
        uri: str,
        **extra: Any,
    ) -> Collection[ResourceReference] | ResourceContent | None: ...


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


def validate_base64(data: str) -> bool:
    try:
        b64decode(data, validate=True)

    except Exception:
        return False

    return True
