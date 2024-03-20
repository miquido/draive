from draive.types.model import Model

__all__ = [
    "ImageContent",
    "ImageBase64Content",
    "ImageURLContent",
]


class ImageURLContent(Model):
    url: str


class ImageBase64Content(Model):
    base64: str


ImageContent = ImageURLContent | ImageBase64Content
