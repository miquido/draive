from draive.parameters.model import DataModel

__all__ = [
    "ImageBase64Content",
    "ImageContent",
    "ImageDataContent",
    "ImageURLContent",
]


class ImageURLContent(DataModel):
    image_url: str
    image_description: str | None = None


class ImageBase64Content(DataModel):
    image_base64: str
    image_description: str | None = None


class ImageDataContent(DataModel):
    image_data: bytes
    image_description: str | None = None


ImageContent = ImageURLContent | ImageBase64Content | ImageDataContent
