from base64 import b64decode, b64encode
from collections.abc import Mapping
from typing import Literal, Self

from draive.multimodal.data_field import b64_or_url_field
from draive.parameters import DataModel

__all__ = [
    "MediaContent",
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


class MediaContent(DataModel):
    @classmethod
    def url(
        cls,
        url: str,
        /,
        mime_type: MediaType,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            mime_type=mime_type,
            source=url,
            meta=meta,
        )

    @classmethod
    def base64(
        cls,
        data: str,
        /,
        mime_type: MediaType,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            mime_type=mime_type,
            source=b64decode(data),
            meta=meta,
        )

    @classmethod
    def data(
        cls,
        data: bytes,
        /,
        mime_type: MediaType,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            mime_type=mime_type,
            source=data,
            meta=meta,
        )

    mime_type: MediaType
    # special field - url string or base64 content auto converted to bytes
    source: str | bytes = b64_or_url_field()
    meta: Mapping[str, str | float | int | bool | None] | None = None

    @property
    def kind(self) -> Literal["image", "audio", "video"]:
        match self.mime_type:
            case "image/jpeg" | "image/png" | "image/bmp" | "image/gif":
                return "image"

            case "audio/aac" | "audio/mpeg" | "audio/ogg" | "audio/wav":
                return "audio"

            case "video/mp4" | "video/mpeg" | "video/ogg":
                return "video"

    def as_string(
        self,
        *,
        include_data: bool,
    ) -> str:
        match self.source:
            case str() as string:
                return f"![{self.kind}({string})"

            case bytes() as data:
                if include_data:
                    return (
                        f"![{self.kind}](data:{self.mime_type};base64,{b64encode(data).decode()})"
                    )

                else:
                    return f"![{self.kind}]()"

    def __bool__(self) -> bool:
        return len(self.source) > 0
