from base64 import b64decode, b64encode
from collections.abc import Sequence
from typing import Final, Literal, Self, cast, get_args

from haiway import Default

from draive.commons import META_EMPTY, Meta
from draive.multimodal.data_field import b64_or_url_field
from draive.parameters import DataModel

__all__ = [
    "MEDIA_KINDS",
    "MediaContent",
    "MediaKind",
    "MediaType",
    "validated_media_kind",
]

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
]
MEDIA_KINDS: Final[Sequence[str]] = get_args(MediaKind)


def validated_media_kind(
    kind: str,
    /,
) -> MediaKind:
    if kind in MEDIA_KINDS:
        return cast(MediaKind, kind)

    else:
        raise ValueError(f"Unsupported media kind: {kind}")


# TODO: split to MediaContent and MediaReferenceContent
class MediaContent(DataModel):
    @classmethod
    def url(
        cls,
        url: str,
        /,
        media: MediaType | MediaKind,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            media=media,
            source=url,
            meta=meta if meta is not None else META_EMPTY,
        )

    @classmethod
    def base64(
        cls,
        data: str,
        /,
        media: MediaType,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            media=media,
            source=b64decode(data),
            meta=meta if meta is not None else META_EMPTY,
        )

    @classmethod
    def data(
        cls,
        data: bytes,
        /,
        media: MediaType,
        meta: Meta | None = None,
    ) -> Self:
        return cls(
            media=media,
            source=data,
            meta=meta if meta is not None else META_EMPTY,
        )

    media: MediaType | MediaKind
    # special field - url string or base64 content auto converted to bytes
    source: str | bytes = b64_or_url_field()
    meta: Meta = Default(META_EMPTY)

    @property
    def kind(self) -> MediaKind:  # noqa: PLR0911
        match self.media:
            case "image" | "image/jpeg" | "image/png" | "image/bmp" | "image/gif":
                return "image"

            case "audio" | "audio/aac" | "audio/mpeg" | "audio/ogg" | "audio/wav":
                return "audio"

            case "video" | "video/mp4" | "video/mpeg" | "video/ogg":
                return "video"

            case other_image if other_image.startswith("image"):
                return "image"

            case other_audio if other_audio.startswith("audio"):
                return "audio"

            case other_video if other_video.startswith("video"):
                return "video"

            case _:
                return "unknown"

    def as_string(
        self,
        *,
        include_data: bool,
    ) -> str:
        match self.source:
            case str() as string:
                return f"![{self.kind}]({string})"

            case bytes() as data:
                if include_data:
                    return f"![{self.kind}](data:{self.media};base64,{b64encode(data).decode()})"

                else:
                    return f"![{self.kind}]()"

    def __bool__(self) -> bool:
        return len(self.source) > 0
