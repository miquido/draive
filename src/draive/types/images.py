from draive.helpers import MISSING, Missing
from draive.types.model import Model

__all__ = [
    "ImageBase64Content",
    "ImageContent",
    "ImageURLContent",
]


class ImageURLContent(Model):
    image_url: str
    image_description: str | Missing = MISSING


class ImageBase64Content(Model):
    image_base64: str
    image_description: str | Missing = MISSING


ImageContent = ImageURLContent | ImageBase64Content
