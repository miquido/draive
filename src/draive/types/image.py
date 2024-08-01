from typing import Literal

from draive.parameters import DataModel

__all__ = [
    "ImageBase64Content",
    "ImageContent",
    "ImageURLContent",
]


class ImageURLContent(DataModel):
    mime_type: Literal["image/jpeg", "image/png", "image/gif"] | None = None
    image_url: str
    image_description: str | None = None
    meta: dict[str, str | float | int | bool | None] | None = None

    def __bool__(self) -> bool:
        return bool(self.image_url)


class ImageBase64Content(DataModel):
    mime_type: Literal["image/jpeg", "image/png", "image/gif"] | None = None
    image_base64: str
    image_description: str | None = None
    meta: dict[str, str | float | int | bool | None] | None = None

    def __bool__(self) -> bool:
        return bool(self.image_base64)


ImageContent = ImageURLContent | ImageBase64Content
