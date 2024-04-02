from draive.generation.image.generator import ImageGenerator
from draive.types import State

__all__ = [
    "ImageGeneration",
]


class ImageGeneration(State):
    generate: ImageGenerator
