from base64 import b64decode, urlsafe_b64encode
from collections.abc import Sequence
from typing import Final, Literal, Self, cast, get_args

from haiway import Default

from draive.commons import META_EMPTY, Meta
from draive.multimodal.data_field import b64_data_field
from draive.parameters import DataModel

__all__ = (
    "MEDIA_KINDS",
    "MediaContent",
    "MediaKind",
    "MediaType",
    "validated_media_kind",
)

MediaType = (
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
    ]
    | str
)


MediaKind = Literal[
    "unknown",
    "image",
    "audio",
    "video",
    "document",
]


class MediaReference(DataModel):
    @classmethod
    def of(
        cls,
        uri: str,
        /,
        media: MediaType | None = None,
        *,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            media=media if media is not None else "unknown",
            uri=uri,
            meta=meta if meta is not None else META_EMPTY,
        )

    media: MediaType
    uri: str
    meta: Meta = Default(META_EMPTY)

    @property
    def kind(self) -> MediaKind:
        return _media_kind(self.media)

    def as_string(self) -> str:
        return f"![{self.kind}]({self.uri})"

    def __bool__(self) -> bool:
        return len(self.uri) > 0


class MediaData(DataModel):
    @classmethod
    def of(
        cls,
        data: str | bytes,
        /,
        media: MediaType,
        *,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            media=media,
            data=data if isinstance(data, bytes) else b64decode(data),
            meta=meta if meta is not None else META_EMPTY,
        )

    media: MediaType
    # special field - base64 content auto converted to bytes
    data: bytes = b64_data_field()
    meta: Meta = Default(META_EMPTY)

    @property
    def kind(self) -> MediaKind:
        return _media_kind(self.media)

    def as_string(
        self,
        *,
        include_data: bool,
    ) -> str:
        if include_data:
            return f"![{self.kind}]({self.as_data_uri()})"

        else:
            return f"![{self.kind}]()"

    def as_data_uri(self) -> str:
        return f"data:{self.media};base64,{urlsafe_b64encode(self.data).decode()}"

    def __bool__(self) -> bool:
        return len(self.data) > 0


MediaContent = MediaReference | MediaData

MEDIA_KINDS: Final[Sequence[str]] = get_args(MediaKind)


def validated_media_kind(
    kind: str,
    /,
) -> MediaKind:
    if kind in MEDIA_KINDS:
        return cast(MediaKind, kind)

    else:
        raise ValueError(f"Unsupported media kind: {kind}")


def _media_kind(
    media: str,
    /,
) -> MediaKind:
    if media.startswith("image"):
        return "image"

    elif media.startswith("audio"):
        return "audio"

    elif media.startswith("video"):
        return "video"

    else:
        return "unknown"
