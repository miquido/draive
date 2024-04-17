from draive.types.model import Model

__all__ = [
    "ImageBase64Content",
    "ImageContent",
    "ImageURLContent",
]


class ImageURLContent(Model):
    image_url: str


class ImageBase64Content(Model):
    image_base64: str


ImageContent = ImageURLContent | ImageBase64Content
