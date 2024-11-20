from base64 import b64decode, b64encode
from collections.abc import Mapping
from typing import Literal, Self

from draive.multimodal.data_field import b64_or_url_field
from draive.parameters import DataModel

__all__ = [
    "MediaContent",
    "MediaKind",
    "MediaType",
]

MediaType = Literal[
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

MediaKind = Literal[
    "image",
    "audio",
    "video",
]


class MediaContent(DataModel):
    @classmethod
    def url(
        cls,
        url: str,
        /,
        media: MediaType | MediaKind,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            media=media,
            source=url,
            meta=meta,
        )

    @classmethod
    def base64(
        cls,
        data: str,
        /,
        media: MediaType,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            media=media,
            source=b64decode(data),
            meta=meta,
        )

    @classmethod
    def data(
        cls,
        data: bytes,
        /,
        media: MediaType,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            media=media,
            source=data,
            meta=meta,
        )

    media: MediaType | MediaKind
    # special field - url string or base64 content auto converted to bytes
    source: str | bytes = b64_or_url_field()
    meta: Mapping[str, str | float | int | bool | None] | None = None

    @property
    def kind(self) -> MediaKind:
        match self.media:
            case "image" | "image/jpeg" | "image/png" | "image/bmp" | "image/gif":
                return "image"

            case "audio" | "audio/aac" | "audio/mpeg" | "audio/ogg" | "audio/wav":
                return "audio"

            case "video" | "video/mp4" | "video/mpeg" | "video/ogg":
                return "video"

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
