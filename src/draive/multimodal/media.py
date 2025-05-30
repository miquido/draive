from base64 import b64decode, b64encode, urlsafe_b64encode
from collections.abc import Sequence
from typing import Final, Literal, Self, cast, get_args

from draive.commons import META_EMPTY, Meta, MetaValues
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
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            media=media if media is not None else "unknown",
            uri=uri,
            meta=Meta.of(meta),
        )

    media: MediaType
    uri: str
    meta: Meta = META_EMPTY

    @property
    def kind(self) -> MediaKind:
        return _media_kind(self.media)

    def to_str(self) -> str:
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
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            media=media,
            data=data if isinstance(data, bytes) else b64decode(data),
            meta=Meta.of(meta),
        )

    media: MediaType
    # special field - base64 content auto converted to bytes
    data: bytes = b64_data_field()
    meta: Meta = META_EMPTY

    @property
    def kind(self) -> MediaKind:
        return _media_kind(self.media)

    def to_str(
        self,
        *,
        include_data: bool = False,
    ) -> str:
        if include_data:
            return f"![{self.kind}]({self.to_data_uri()})"

        else:
            return f"![{self.kind}]()"

    def to_data_uri(
        self,
        *,
        safe_encoding: bool = True,
    ) -> str:
        encoded: str
        if safe_encoding:
            encoded = urlsafe_b64encode(self.data).decode("utf-8")

        else:
            encoded = b64encode(self.data).decode("utf-8")

        return f"data:{self.media};base64,{encoded}"

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
