from draive.generation.image.generator import ImageGenerator
from draive.parameters import State

__all__ = [
    "ImageGeneration",
]


class ImageGeneration(State):
    generate: ImageGenerator
