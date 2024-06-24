from typing import Literal

from draive.parameters.model import DataModel

__all__ = [
    "ImageBase64Content",
    "ImageContent",
    "ImageDataContent",
    "ImageURLContent",
]


class ImageURLContent(DataModel):
    mime_type: Literal["image/jpeg", "image/png", "image/gif"] | None = None
    image_url: str
    image_description: str | None = None


class ImageBase64Content(DataModel):
    mime_type: Literal["image/jpeg", "image/png", "image/gif"] | None = None
    image_base64: str
    image_description: str | None = None


class ImageDataContent(DataModel):
    mime_type: Literal["image/jpeg", "image/png", "image/gif"] | None = None
    image_data: bytes
    image_description: str | None = None


ImageContent = ImageURLContent | ImageBase64Content | ImageDataContent
