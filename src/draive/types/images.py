from draive.types.model import Model

__all__ = [
    "ImageBase64Content",
    "ImageContent",
    "ImageURLContent",
]


class ImageURLContent(Model):
    image_url: str
    image_description: str | None = None


class ImageBase64Content(Model):
    image_base64: str
    image_description: str | None = None


ImageContent = ImageURLContent | ImageBase64Content
